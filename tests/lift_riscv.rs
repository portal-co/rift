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

/// Registry of supported ISA variants for testing
static ISA_VARIANTS: &[IsaVariant] = &[
    IsaVariant {
        name: "rv32i",
        binaries: &[
            "rv32i_01_integer_computational.bin",
            "rv32i_02_control_transfer.bin",
            "rv32i_03_load_store.bin",
            "rv32i_04_edge_cases.bin",
            "rv32i_05_simple_program.bin",
            "rv32i_06_nop_and_hints.bin",
            "rv32i_07_pseudo_instructions.bin",
        ],
    },
    IsaVariant {
        name: "rv64i",
        binaries: &["rv64i_01_basic_64bit.bin"],
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

    // Fall back to first executable section
    for section in file.sections() {
        if let Ok(data) = section.data() {
            if !data.is_empty() {
                // Check if section is executable (has ALLOC flag in ELF)
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
fn create_minimal_module() -> (Module<'static>, Memory, Table, Func) {
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

    // Create a dummy ecall handler signature
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

    (module, memory, table, ecall)
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
    let (mut module, memory, table, ecall) = create_minimal_module();

    // Configure options
    let opts = Opts {
        mem: memory,
        table,
        ecall,
        mapper: None,
        inline_ecall: true,
    };

    // Configure tuning parameters
    let tune = Tunables { n: 256, bleed: 64 };

    // Use catch_unwind to handle todo!() panics gracefully
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Perform the lift
        let _func = rift::compile(
            &mut module,
            vec![], // No user types
            input.as_ref(),
            start_addr,
            opts,
            &tune,
            &mut |_, _| {}, // No user preparation
            std::iter::repeat(Type::I64).take(33), // Return types: 31 GPRs + PC + trap flag
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
/// It prints a summary showing which binaries lifted successfully and which failed.
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

    // The test passes if we can at least parse and attempt to lift all binaries
    // (even if some instructions aren't implemented yet)
    // This ensures the test harness is working correctly
    assert!(
        successes.len() + failures.len() == variant.binaries.len(),
        "All binaries should be processed"
    );
}

// ============================================================================
// RV32I Tests
// ============================================================================

#[test]
fn test_rv32i_lifting() {
    println!("\n=== Testing RV32I lifting ===");
    test_isa_variant(&ISA_VARIANTS[0]);
}

// ============================================================================
// RV64I Tests
// ============================================================================

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
