//! Integration tests for lifting RISC-V binaries to WebAssembly
//!
//! This test suite uses binaries from the rv-corpus repository:
//! https://github.com/portal-co/rv-corpus
//!
//! Tests are organized by ISA variant (rv32i, rv64i) with an extensible
//! structure for adding more variants as support is added.

use bitvec::prelude::*;
use object::{Object, ObjectSection};
use portal_pc_asm_common::types::{Input, Perms};
use portal_pc_waffle::{Func, FuncDecl, Memory, MemoryData, Module, Table, TableData, Type, HeapType};
use portal_pc_waffle::passes::reorder_funs::fixup_orders;
use rift::{Opts, Tunables};
use std::path::Path;

/// Configuration for a RISC-V ISA variant
#[derive(Clone, Debug)]
struct IsaVariant {
    /// Name of the variant (e.g., "rv32i", "rv64i")
    name: &'static str,
    /// Test binary files for this variant (relative to fixtures directory)
    binaries: &'static [&'static str],
}

/// Minimal WebAssembly module infrastructure for lifting RISC-V code
struct MinimalModule {
    /// The WebAssembly module being constructed
    module: Module<'static>,
    /// Memory for the lifted code to use
    memory: Memory,
    /// Table for indirect calls
    table: Table,
    /// The ecall handler function import
    ecall: Func,
}

/// Registry of supported ISA variants for testing
/// 
/// All known test binaries are included here. Tests that fail due to
/// unimplemented instructions are documented in test_isa_variant.
static ISA_VARIANTS: &[IsaVariant] = &[
    IsaVariant {
        name: "rv32i",
        binaries: &[
            "rv32i/01_integer_computational",
            "rv32i/02_control_transfer",
            "rv32i/03_load_store",
            "rv32i/04_edge_cases",
            "rv32i/05_simple_program",
            "rv32i/06_nop_and_hints",
            "rv32i/07_pseudo_instructions",
        ],
    },
    IsaVariant {
        name: "rv64i",
        binaries: &["rv64i/01_basic_64bit"],
    },
];

/// Extract executable code from an ELF file
fn extract_code_from_elf(data: &[u8]) -> Option<(Vec<u8>, u64)> {
    let file = object::File::parse(data).ok()?;

    // Try to find the .text section first
    if let Some(section) = file.section_by_name(".text") {
        let code = section.data().ok()?;
        let start_addr = section.address();
        return Some((code.to_vec(), start_addr));
    }

    // Fall back to first non-empty section with data
    for section in file.sections() {
        if let Ok(data) = section.data() {
            if !data.is_empty() {
                return Some((data.to_vec(), section.address()));
            }
        }
    }

    None
}

/// Create an Input structure from raw code bytes
fn create_input(code: Vec<u8>) -> Input {
    let len = code.len();
    let perms = Perms {
        r: bitvec![1; len],
        w: bitvec![0; len],
        x: bitvec![1; len],
        nj: bitvec![0; len],
    };
    Input::new(code, perms).expect("Failed to create Input")
}

/// Create a minimal module with required infrastructure for lifting
fn create_minimal_module() -> MinimalModule {
    let mut module = Module::empty();

    // Add a memory for the lifted code to use
    let memory = module.memories.push(MemoryData {
        initial_pages: 1,
        maximum_pages: Some(256),
        shared: false,
        memory64: false,
        page_size_log2: None,
        segments: vec![],
    });

    // Add a table for indirect calls
    let table = module.tables.push(TableData {
        ty: Type::Heap(portal_pc_waffle::WithNullable {
            value: HeapType::FuncRef,
            nullable: true,
        }),
        initial: 1024,
        max: Some(4096),
        func_elements: Some(vec![]),
        table64: false,
    });

    // Create ecall handler signature with 32 params/returns for RISC-V registers
    // RISC-V has 32 GPRs (x0-x31), though x0 is hardwired to zero
    let ecall_sig = portal_pc_waffle::util::new_sig(
        &mut module,
        portal_pc_waffle::SignatureData::Func {
            params: std::iter::repeat(Type::I64).take(32).collect(),
            returns: std::iter::repeat(Type::I64).take(32).collect(),
            shared: true,
        },
    );

    // Create an import for the ecall handler
    let ecall = module.funcs.push(FuncDecl::Import(
        ecall_sig,
        "ecall_handler".to_string(),
    ));

    // Register in the imports list - this is required for the waffle backend
    // to correctly count imports when generating the function section
    module.imports.push(portal_pc_waffle::Import {
        module: "env".to_string(),
        name: "ecall_handler".to_string(),
        kind: portal_pc_waffle::ImportKind::Func(ecall),
    });

    MinimalModule { module, memory, table, ecall }
}

/// Lift a RISC-V binary and verify the output WebAssembly is valid
/// 
/// Returns Ok(()) if lifting succeeds, or Err with a descriptive message.
/// Note: Some instructions may not be implemented yet (todo!() in lib.rs),
/// which will cause this function to return an error.
fn lift_and_validate(binary_path: &Path) -> Result<(), String> {
    // Read the ELF file
    let data = std::fs::read(binary_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Extract code from ELF
    let (code, start_addr) = extract_code_from_elf(&data)
        .ok_or_else(|| "Failed to extract code from ELF".to_string())?;

    if code.is_empty() {
        return Err("Empty code section".to_string());
    }

    // Create the input structure
    let input = create_input(code);

    // Create a minimal module
    let MinimalModule { mut module, memory, table, ecall } = create_minimal_module();

    // Configure options
    let opts = Opts {
        mem: memory,
        table,
        ecall,
        mapper: None,
        inline_ecall: true,
        process_hints: false, // Disable HINT processing for basic lifting tests
    };

    // Configure tuning parameters
    let tune = Tunables { n: 256, bleed: 64 };

    // Use catch_unwind to handle todo!() panics gracefully
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Perform the lift
        // Return types: 31 values for GPRs x1-x31 (x0 is always zero) + PC + trap flag = 33
        let _func = rift::compile(
            &mut module,
            vec![], // No user types
            input.as_ref(),
            start_addr,
            opts,
            &tune,
            &mut |_, _| {}, // No user preparation
            std::iter::repeat(Type::I64).take(33),
        );

        // Use fixup_orders to ensure imports come before function bodies
        // This is required for valid WebAssembly module structure
        fixup_orders(&mut module);

        // Validate by serializing to WebAssembly bytes
        module.to_wasm_bytes()
    }));

    match result {
        Ok(Ok(wasm_bytes)) => {
            // Basic validation: check WASM magic number
            if wasm_bytes.len() < 8 {
                return Err("Generated WebAssembly is too small".to_string());
            }

            // WASM magic number: \0asm
            if &wasm_bytes[0..4] != b"\0asm" {
                return Err("Invalid WebAssembly magic number".to_string());
            }

            Ok(())
        }
        Ok(Err(e)) => Err(format!("Failed to serialize to WebAssembly: {:?}", e)),
        Err(_) => Err("Panic during lifting (likely unimplemented instruction)".to_string()),
    }
}

/// Test all binaries for a specific ISA variant
/// 
/// This function tests each binary and tracks successes/failures.
/// Tests will fail if any binary fails to lift, providing clear diagnostics
/// about which instructions or features are missing.
/// 
/// # Current Known Failures
/// 
/// Most test binaries currently fail due to unimplemented instructions in the lifter.
/// The `todo!()` macro in `src/lib.rs:512` is hit when encountering unsupported
/// RISC-V instructions. As more instructions are implemented, more tests will pass.
/// 
/// ## RV32I Status
/// - `01_integer_computational`: Fails - unimplemented instruction
/// - `02_control_transfer`: Fails - unimplemented instruction  
/// - `03_load_store`: Fails - unimplemented instruction
/// - `04_edge_cases`: Fails - unimplemented instruction
/// - `05_simple_program`: Fails - arithmetic overflow in lifter
/// - `06_nop_and_hints`: **PASSES** - only uses NOP/hint instructions
/// - `07_pseudo_instructions`: Fails - unimplemented instruction
/// 
/// ## RV64I Status
/// - `01_basic_64bit`: Fails - unimplemented instruction
fn test_isa_variant(variant: &IsaVariant) {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");

    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for binary in variant.binaries {
        let binary_path = fixtures_dir.join(binary);

        if !binary_path.exists() {
            failures.push(format!("{}: file not found", binary));
            continue;
        }

        match lift_and_validate(&binary_path) {
            Ok(()) => {
                println!("  ✓ {}", binary);
                successes.push(binary.to_string());
            }
            Err(e) => {
                println!("  ✗ {}: {}", binary, e);
                failures.push(format!("{}: {}", binary, e));
            }
        }
    }

    println!("\nSummary for {}:", variant.name);
    println!("  Passed: {}/{}", successes.len(), variant.binaries.len());
    println!("  Failed: {}/{}", failures.len(), variant.binaries.len());

    // Fail the test if any binaries failed to lift
    // This makes test failures visible without requiring --nocapture
    assert!(
        failures.is_empty(),
        "\n{} test(s) failed for {}:\n  {}\n\nSee doc comments on test_isa_variant for known failure reasons.",
        failures.len(),
        variant.name,
        failures.join("\n  ")
    );
}

// ============================================================================
// RV32I Tests
// ============================================================================

/// RV32I lifting test - tests all rv32i binaries from rv-corpus.
/// 
/// Currently most tests fail due to unimplemented instructions.
/// See test_isa_variant documentation for detailed failure reasons.
#[test]
fn test_rv32i_lifting() {
    println!("\n=== Testing RV32I lifting ===");
    test_isa_variant(&ISA_VARIANTS[0]);
}

// ============================================================================
// RV64I Tests
// ============================================================================

/// RV64I lifting test - tests all rv64i binaries from rv-corpus.
/// 
/// Currently all tests fail due to unimplemented instructions.
/// See test_isa_variant documentation for detailed failure reasons.
#[test]
fn test_rv64i_lifting() {
    println!("\n=== Testing RV64I lifting ===");
    test_isa_variant(&ISA_VARIANTS[1]);
}

// ============================================================================
// Extensibility Tests
// ============================================================================

/// This test verifies the test infrastructure is extensible
/// by checking that all registered variants can be iterated
#[test]
fn test_isa_variants_registered() {
    assert!(
        !ISA_VARIANTS.is_empty(),
        "At least one ISA variant should be registered"
    );

    // Count total binaries
    let total: usize = ISA_VARIANTS.iter().map(|v| v.binaries.len()).sum();
    assert!(total > 0, "At least one binary should be registered");

    println!("\nRegistered ISA variants:");
    for variant in ISA_VARIANTS {
        println!(
            "  {} ({} binaries)",
            variant.name,
            variant.binaries.len()
        );
    }
}

/// Test that we can add a new ISA variant at runtime (for extensibility demonstration)
#[test]
fn test_extensibility_pattern() {
    // Demonstrate the pattern for adding new variants
    let custom_variant = IsaVariant {
        name: "rv32im",
        binaries: &[], // Would contain binaries when M extension is supported
    };

    // Verify the structure works
    assert_eq!(custom_variant.name, "rv32im");
    assert!(custom_variant.binaries.is_empty());
}

// ============================================================================
// HINT Detection Tests
// ============================================================================

/// Test that detect_hint correctly identifies HINT instructions from rv-corpus
#[test]
fn test_detect_hint_basic() {
    use rv_asm::{Inst, Reg, Imm};
    
    // HINT 1: addi x0, x0, 1
    let hint1 = Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(1),
    };
    assert_eq!(rift::detect_hint(&hint1), Some(1));
    
    // HINT 42: addi x0, x0, 42
    let hint42 = Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(42),
    };
    assert_eq!(rift::detect_hint(&hint42), Some(42));
    
    // HINT with negative marker (allowed by format)
    let hint_neg = Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(-5),
    };
    assert_eq!(rift::detect_hint(&hint_neg), Some(-5));
    
    // NOP (addi x0, x0, 0) should NOT be detected as a test marker
    let nop = Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(0),
    };
    assert_eq!(rift::detect_hint(&nop), None);
    
    // Regular ADDI (not to x0) should not be detected
    let regular_addi = Inst::Addi {
        dest: Reg(10), // a0
        src1: Reg(0),
        imm: Imm::new_i32(5),
    };
    assert_eq!(rift::detect_hint(&regular_addi), None);
    
    // ADDI with rs1 != x0 should not be detected as primary HINT format
    let hint_other_src = Inst::Addi {
        dest: Reg(0),
        src1: Reg(10), // a0
        imm: Imm::new_i32(0),
    };
    assert_eq!(rift::detect_hint(&hint_other_src), None);
}

/// Test that is_hint_instruction correctly identifies all HINT types
#[test]
fn test_is_hint_instruction() {
    use rv_asm::{Inst, Reg, Imm};
    
    // All these should be identified as HINTs (rd=x0)
    
    // ADDI x0, x0, 0 (NOP) is a HINT
    assert!(rift::is_hint_instruction(&Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(0),
    }));
    
    // ADDI x0, x0, N is a HINT
    assert!(rift::is_hint_instruction(&Inst::Addi {
        dest: Reg(0),
        src1: Reg(0),
        imm: Imm::new_i32(5),
    }));
    
    // ADDI x0, rs1, 0 is a HINT
    assert!(rift::is_hint_instruction(&Inst::Addi {
        dest: Reg(0),
        src1: Reg(10),
        imm: Imm::new_i32(0),
    }));
    
    // ADD x0, rs1, rs2 is a HINT
    assert!(rift::is_hint_instruction(&Inst::Add {
        dest: Reg(0),
        src1: Reg(10),
        src2: Reg(11),
    }));
    
    // Regular ADD (rd != x0) is NOT a HINT
    assert!(!rift::is_hint_instruction(&Inst::Add {
        dest: Reg(10),
        src1: Reg(11),
        src2: Reg(12),
    }));
    
    // Load instructions are NOT HINTs even if dest is x0
    // (they have side effects beyond the register write)
    assert!(!rift::is_hint_instruction(&Inst::Lw {
        dest: Reg(0),
        base: Reg(2),
        offset: Imm::new_i32(0),
    }));
}

/// Test scan_hints on actual rv-corpus binary
#[test]
fn test_scan_hints_on_binary() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    
    let binary_path = fixtures_dir.join("rv32i/01_integer_computational");
    
    if !binary_path.exists() {
        println!("Skipping test: binary not found (submodule not initialized?)");
        return;
    }
    
    let data = std::fs::read(&binary_path).expect("Failed to read binary");
    let (code, start_addr) = extract_code_from_elf(&data).expect("Failed to extract code");
    
    // Scan for HINTs
    let hints = rift::scan_hints(&code, start_addr, rv_asm::Xlen::Rv32);
    
    // The 01_integer_computational.s file has HINT markers 1-14
    // Verify we found some HINTs
    assert!(!hints.is_empty(), "Should find HINT markers in test binary");
    
    // Check that marker values are reasonable (1-14 based on the source)
    for hint in &hints {
        // rv-corpus uses markers 1-2047 for positive markers
        assert!(hint.marker > 0 && hint.marker <= 2047, 
            "Marker {} should be in valid range", hint.marker);
    }
    
    // Verify we found the expected number of markers (14 in this file)
    assert_eq!(hints.len(), 14, 
        "Expected 14 HINT markers in 01_integer_computational, found {}", hints.len());
}

/// Test that HintInfo struct works correctly
#[test]
fn test_hint_info_struct() {
    let hint = rift::HintInfo {
        marker: 42,
        pc: 0x1000,
    };
    
    assert_eq!(hint.marker, 42);
    assert_eq!(hint.pc, 0x1000);
    
    // Test equality
    let hint2 = rift::HintInfo {
        marker: 42,
        pc: 0x1000,
    };
    assert_eq!(hint, hint2);
    
    let hint3 = rift::HintInfo {
        marker: 43,
        pc: 0x1000,
    };
    assert_ne!(hint, hint3);
}

/// Test scan_all_hints returns both markers and non-marker HINTs
#[test]
fn test_scan_all_hints() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    
    let binary_path = fixtures_dir.join("rv32i/06_nop_and_hints");
    
    if !binary_path.exists() {
        println!("Skipping test: binary not found (submodule not initialized?)");
        return;
    }
    
    let data = std::fs::read(&binary_path).expect("Failed to read binary");
    let (code, start_addr) = extract_code_from_elf(&data).expect("Failed to extract code");
    
    // Scan for all HINTs (including NOPs)
    let all_hints = rift::scan_all_hints(&code, start_addr, rv_asm::Xlen::Rv32);
    
    // Should find multiple HINTs/NOPs in this file
    assert!(!all_hints.is_empty(), "Should find HINTs in test binary");
    
    let mut markers = 0;
    let mut nops = 0;
    
    for (pc, inst, marker) in &all_hints {
        if let Some(m) = marker {
            println!("Found HINT marker {} at PC {:#x}: {}", m, pc, inst);
            markers += 1;
        } else {
            println!("Found NOP/HINT at PC {:#x}: {}", pc, inst);
            nops += 1;
        }
    }
    
    println!("Total: {} markers, {} NOPs/other HINTs", markers, nops);
    
    // The 06_nop_and_hints.s file has both NOPs and HINT instructions
    assert!(nops > 0, "Should find NOPs in nop_and_hints test");
}

/// Test compile_with_hints callback is invoked for HINT instructions
#[test]
fn test_compile_with_hints_callback() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    
    let binary_path = fixtures_dir.join("rv32i/01_integer_computational");
    
    if !binary_path.exists() {
        println!("Skipping test: binary not found (submodule not initialized?)");
        return;
    }
    
    let data = std::fs::read(&binary_path).expect("Failed to read binary");
    let (code, start_addr) = extract_code_from_elf(&data).expect("Failed to extract code");
    
    // Create the input structure
    let input = create_input(code);
    
    // Create a minimal module
    let MinimalModule { mut module, memory, table, ecall } = create_minimal_module();
    
    // Configure options with HINT processing enabled
    let opts = Opts {
        mem: memory,
        table,
        ecall,
        mapper: None,
        inline_ecall: true,
        process_hints: true, // Enable HINT processing
    };
    
    // Configure tuning parameters
    let tune = Tunables { n: 256, bleed: 64 };
    
    // Collect HINTs via callback
    let mut collected_hints: Vec<rift::HintInfo> = Vec::new();
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _func = rift::compile_with_hints(
            &mut module,
            vec![], // No user types
            input.as_ref(),
            start_addr,
            opts,
            &tune,
            &mut |_, _| {}, // No user preparation
            std::iter::repeat(Type::I64).take(33),
            &mut |ctx| {
                // Callback is invoked for each HINT
                collected_hints.push(ctx.hint.clone());
            },
        );
    }));
    
    match result {
        Ok(()) => {
            // Verify that the callback collected the expected HINTs
            assert!(!collected_hints.is_empty(), "Should collect HINT markers via callback");
            
            // The 01_integer_computational.s file has HINT markers 1-14
            assert_eq!(collected_hints.len(), 14, 
                "Expected 14 HINT markers, found {}", collected_hints.len());
            
            // Verify marker values are in expected range
            for hint in &collected_hints {
                assert!(hint.marker >= 1 && hint.marker <= 14,
                    "Marker {} should be between 1 and 14", hint.marker);
            }
            
            // Verify we got markers 1-14 (in any order due to compilation order)
            let mut markers: Vec<i32> = collected_hints.iter().map(|h| h.marker).collect();
            markers.sort();
            let expected: Vec<i32> = (1..=14).collect();
            assert_eq!(markers, expected, "Should have markers 1-14");
        }
        Err(e) => {
            panic!("compile_with_hints should not panic: {:?}", e);
        }
    }
}
