#![no_std]
extern crate alloc;
use alloc::format;
use alloc::vec::Vec;
use alloc::{boxed::Box, vec};
use core::{
    iter::once,
    mem::{replace, swap},
    ops::Not,
    u64,
};
use portal_pc_waffle::SignatureData;
use rv_asm::{FReg, Inst, Reg};
// use std::env::consts::FAMILY;
// use crate::{Regs, Tunables, Utils};
// use enum_map::Enum;
use itertools::Itertools;
use portal_pc_asm_common::types::InputRef;
use portal_pc_waffle::{
    Block, BlockTarget, Func, FuncDecl, FunctionBody, Memory, MemoryArg, Module, Operator, Table,
    Type, Value, util::new_sig, util::results_ref_2,
};

/// Information about a detected HINT instruction.
///
/// HINT instructions are RISC-V instructions that write to x0 (which is hardwired
/// to zero), making them architectural no-ops that can carry metadata.
///
/// The primary format used by rv-corpus is `addi x0, x0, N` where N is a test
/// case marker value (1-2047 for positive values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HintInfo {
    /// The immediate value from the HINT instruction (marker value).
    /// For `addi x0, x0, N`, this is N.
    pub marker: i32,
    /// The PC address where this HINT was encountered.
    pub pc: u64,
}

/// Context passed to the HINT callback during inline processing.
///
/// This structure provides access to the current compilation state when a HINT
/// instruction is encountered, allowing the callback to inspect or modify the
/// generated WebAssembly code.
///
/// For complex branching, the handler can modify the `block` field to change
/// which block subsequent instructions will be added to.
pub struct HintCallbackContext<'a, 'b> {
    /// The HINT information (marker value and PC)
    pub hint: HintInfo,
    /// The decoded HINT instruction
    pub instruction: &'a Inst,
    /// The WebAssembly module being constructed
    pub module: &'a mut Module<'b>,
    /// The current function body being built
    pub function: &'a mut FunctionBody,
    /// The current block where the HINT was encountered.
    /// This can be modified by the handler to support complex branching -
    /// subsequent code will be added to the updated block.
    pub block: &'a mut Block,
    /// The current register state
    pub regs: &'a mut Regs,
    /// The PC value as a WebAssembly Value
    pub pc_value: Value,
}

/// Trait for handling HINT instructions during compilation.
///
/// This trait is invoked for each HINT instruction encountered during
/// compilation when a handler is provided to `compile_with_hints`.
///
/// The handler receives a `HintCallbackContext` which provides access to:
/// - The HINT marker value and PC
/// - The decoded instruction
/// - The WebAssembly module and function being built
/// - The current block and register state
///
/// # Example
///
/// ```ignore
/// struct MyHintHandler {
///     hints_found: Vec<HintInfo>,
/// }
///
/// impl HintHandler for MyHintHandler {
///     fn on_hint(&mut self, ctx: &mut HintCallbackContext<'_, '_>) {
///         self.hints_found.push(ctx.hint.clone());
///     }
/// }
/// ```
pub trait HintHandler {
    /// Called when a HINT instruction is encountered during compilation.
    ///
    /// The context provides mutable access to the compilation state, allowing
    /// the handler to inspect the HINT or modify the generated WebAssembly code.
    fn on_hint(&mut self, ctx: &mut HintCallbackContext<'_, '_>);
}

/// Blanket implementation of `HintHandler` for any `FnMut` closure.
///
/// This allows using closures directly as hint handlers:
///
/// ```ignore
/// let mut hints_found = Vec::new();
/// compile_with_hints(
///     // ... other args ...
///     &mut |ctx: &mut HintCallbackContext| {
///         hints_found.push(ctx.hint.clone());
///     },
/// );
/// ```
impl<F> HintHandler for F
where
    F: FnMut(&mut HintCallbackContext<'_, '_>),
{
    fn on_hint(&mut self, ctx: &mut HintCallbackContext<'_, '_>) {
        self(ctx)
    }
}

/// Checks if an instruction is a HINT instruction.
///
/// According to the RISC-V specification (Section 2.9):
/// "HINT instructions are usually used to communicate performance hints to the
/// microarchitecture. HINTs are encoded as integer computational instructions
/// with rd=x0."
///
/// This function specifically detects the `addi x0, x0, N` format used by
/// rv-corpus for test case markers.
///
/// Returns `Some(marker)` if the instruction is a HINT with a non-zero
/// immediate (since `addi x0, x0, 0` is the canonical NOP encoding).
pub fn detect_hint(inst: &Inst) -> Option<i32> {
    match inst {
        // Primary HINT format: addi x0, x0, imm (where imm != 0)
        // When imm == 0, this is the canonical NOP, not a test marker
        Inst::Addi { dest, src1, imm } if dest.0 == 0 && src1.0 == 0 => {
            let marker = imm.as_i32();
            if marker != 0 {
                Some(marker)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Checks if an instruction is any kind of HINT (including NOP).
///
/// This is a broader check that identifies any instruction that writes to x0,
/// which includes:
/// - `addi x0, x0, 0` (NOP)
/// - `addi x0, x0, N` (HINT with marker)
/// - `addi x0, rs1, 0` (HINT)
/// - Other computational instructions with rd=x0
pub fn is_hint_instruction(inst: &Inst) -> bool {
    match inst {
        // Any ADDI with dest=x0 is a HINT
        Inst::Addi { dest, .. } if dest.0 == 0 => true,
        // Other computational instructions with rd=x0 are also HINTs
        Inst::Slti { dest, .. } if dest.0 == 0 => true,
        Inst::Sltiu { dest, .. } if dest.0 == 0 => true,
        Inst::Andi { dest, .. } if dest.0 == 0 => true,
        Inst::Ori { dest, .. } if dest.0 == 0 => true,
        Inst::Xori { dest, .. } if dest.0 == 0 => true,
        Inst::Slli { dest, .. } if dest.0 == 0 => true,
        Inst::Srli { dest, .. } if dest.0 == 0 => true,
        Inst::Srai { dest, .. } if dest.0 == 0 => true,
        Inst::Add { dest, .. } if dest.0 == 0 => true,
        Inst::Sub { dest, .. } if dest.0 == 0 => true,
        Inst::Sll { dest, .. } if dest.0 == 0 => true,
        Inst::Srl { dest, .. } if dest.0 == 0 => true,
        Inst::Sra { dest, .. } if dest.0 == 0 => true,
        Inst::Slt { dest, .. } if dest.0 == 0 => true,
        Inst::Sltu { dest, .. } if dest.0 == 0 => true,
        Inst::And { dest, .. } if dest.0 == 0 => true,
        Inst::Or { dest, .. } if dest.0 == 0 => true,
        Inst::Xor { dest, .. } if dest.0 == 0 => true,
        // LUI and AUIPC with non-zero immediate and rd=x0 are also HINTs
        Inst::Lui { dest, uimm } if dest.0 == 0 && uimm.as_u32() != 0 => true,
        Inst::Auipc { dest, uimm } if dest.0 == 0 && uimm.as_u32() != 0 => true,
        _ => false,
    }
}

/// Register state for RISC-V execution.
///
/// Contains both integer registers (GPRs) and floating-point registers (FPRs).
///
/// RISC-V Specification Quote:
/// "The F extension adds 32 floating-point registers, f0–f31, each 32 bits wide,
/// and a floating-point control and status register fcsr, which contains the
/// operating mode and exception status of the floating-point unit."
pub struct Regs {
    /// General purpose registers x1-x31 (x0 is hardwired to zero)
    pub gpr: [Value; 31],
    /// Floating-point registers f0-f31
    ///
    /// RISC-V Specification Quote:
    /// "The D extension widens the 32 floating-point registers, f0–f31, to 64 bits."
    /// 
    /// We store all FPRs as F64 (64-bit) to support both F and D extensions.
    /// For single-precision values, the upper 32 bits are NaN-boxed as per the spec.
    pub fpr: [Value; 32],
    pub user: Vec<Value>,
}
impl Regs {
    pub fn from_args<T>(
        user: &[T],
        mut args: &mut (dyn Iterator<Item = Value> + '_),
    ) -> Option<Self> {
        Some(Regs {
            user: user.iter().filter_map(|_x| args.next()).collect(),
            gpr: {
                let mut args: &mut &mut _ = &mut args;
                args.next_array()?
            },
            fpr: {
                let mut args: &mut &mut _ = &mut args;
                args.next_array()?
            },
        })
    }
    pub fn to_args(&self) -> impl Iterator<Item = Value> {
        return self.user.iter().chain(self.gpr.iter()).chain(self.fpr.iter()).cloned();
    }
    pub fn get(&self, f: &mut FunctionBody, block: Block, Reg(mut i): Reg) -> Value {
        i = i % 32;
        if i == 0 {
            f.add_op(block, Operator::I64Const { value: 0 }, &[], &[Type::I64])
        } else {
            self.gpr[(i - 1) as usize]
        }
    }
    pub fn put(&mut self, Reg(mut i): Reg, v: Value) {
        i = i % 32;
        if i == 0 {
            return;
        }
        i -= 1;
        self.gpr[i as usize] = v;
    }
    pub fn pop(&mut self, f: &mut FunctionBody, block: Block, Reg(mut i): Reg, v: Value) -> Value {
        i = i % 32;
        if i == 0 {
            return f.add_op(block, Operator::I64Const { value: 0 }, &[], &[Type::I64]);
        }
        i -= 1;
        replace(&mut self.gpr[i as usize], v)
    }
    
    /// Get a floating-point register value.
    ///
    /// RISC-V Specification Quote:
    /// "The F extension adds 32 floating-point registers, f0–f31, each 32 bits wide."
    /// "The D extension widens the 32 floating-point registers, f0–f31, to 64 bits."
    ///
    /// Unlike integer registers, f0 is NOT hardwired to zero.
    pub fn get_f(&self, FReg(mut i): FReg) -> Value {
        i = i % 32;
        self.fpr[i as usize]
    }
    
    /// Set a floating-point register value.
    pub fn put_f(&mut self, FReg(mut i): FReg, v: Value) {
        i = i % 32;
        self.fpr[i as usize] = v;
    }
    
    /// Exchange a floating-point register value, returning the old value.
    pub fn pop_f(&mut self, FReg(mut i): FReg, v: Value) -> Value {
        i = i % 32;
        replace(&mut self.fpr[i as usize], v)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tunables {
    pub n: usize,
    pub bleed: usize,
}
impl Tunables {
    pub fn paged_chunks<T: Clone>(
        &self,
        pages: impl Iterator<Item = T> + Clone,
    ) -> impl Iterator<Item: Iterator<Item = T> + Clone> + Clone {
        let n = self.n;
        let bleed = self.bleed;
        let pages = &pages.chunks(n);
        let pages = pages.into_iter().map(|a| a.collect_vec()).collect_vec();
        (0..pages.len()).map(move |i| {
            let page = &pages[i];
            return (0..=(bleed / n))
                .rev()
                .flat_map(|j| pages.get(i.wrapping_sub(j + 1)))
                .flat_map(|a| a.iter())
                .skip(if i * n > bleed {
                    (bleed / (i * n)).saturating_sub(bleed)
                } else {
                    0usize
                })
                .chain(page.iter())
                .chain(
                    (0..=(bleed / n))
                        .flat_map(|j| pages.get(i + j + 1).into_iter().flat_map(|a| a.iter()))
                        .take(bleed),
                )
                .cloned()
                .collect_vec()
                .into_iter();
        })
    }
}

/// NaN-box a single-precision floating-point value for storage in a 64-bit FPR.
/// 
/// RISC-V Specification Quote:
/// "When multiple floating-point precisions are supported, then valid values of narrower n-bit
/// types, n < FLEN, are represented in the lower n bits of an FLEN-bit NaN value, in a process
/// termed NaN-boxing. The upper bits of a valid NaN-boxed value must be all 1s."
/// 
/// Implementation: F32 -> I32 (reinterpret) -> I64 (zero extend) -> OR with 0xFFFFFFFF_00000000 -> F64 (reinterpret)
fn nan_box_f32(f: &mut FunctionBody, block: Block, val: Value) -> Value {
    // Reinterpret F32 as I32
    let i32_val = f.add_op(block, Operator::I32ReinterpretF32, &[val], &[Type::I32]);
    // Zero-extend I32 to I64
    let i64_val = f.add_op(block, Operator::I64ExtendI32U, &[i32_val], &[Type::I64]);
    // OR with upper 32 bits set to all 1s (NaN-boxing)
    let nan_box_mask = f.add_op(block, Operator::I64Const { value: 0xFFFFFFFF_00000000 }, &[], &[Type::I64]);
    let boxed = f.add_op(block, Operator::I64Or, &[i64_val, nan_box_mask], &[Type::I64]);
    // Reinterpret as F64
    f.add_op(block, Operator::F64ReinterpretI64, &[boxed], &[Type::F64])
}

/// Unbox a single-precision floating-point value from a 64-bit FPR.
/// 
/// RISC-V Specification Quote:
/// "If the value is not a valid NaN-boxed value, the value is treated as if it were the
/// canonical quiet NaN."
/// 
/// Implementation: F64 -> I64 (reinterpret) -> I32 (wrap/extract low 32 bits) -> F32 (reinterpret)
/// Note: For simplicity, we extract the lower 32 bits directly. A full implementation would
/// check if the upper 32 bits are all 1s and return canonical NaN if not.
fn nan_unbox_to_f32(f: &mut FunctionBody, block: Block, val: Value) -> Value {
    // Reinterpret F64 as I64
    let i64_val = f.add_op(block, Operator::I64ReinterpretF64, &[val], &[Type::I64]);
    // Wrap/extract lower 32 bits (I64 -> I32)
    let i32_val = f.add_op(block, Operator::I32WrapI64, &[i64_val], &[Type::I32]);
    // Reinterpret as F32
    f.add_op(block, Operator::F32ReinterpretI32, &[i32_val], &[Type::F32])
}

#[derive(Clone, Copy)]
struct RiscVContext {
    block: Block,
    // mem: Memory,
    rbs: [Block; 8193],
    pc_value: Value,
    original_pc_value: Value,
    pc: u64,
    local_instruction_index: usize,
    opts: Opts,
    // ecall: Func,
    // table: Table,
}
fn compile_one(
    module: &mut Module,
    f: &mut FunctionBody,
    r: &mut Regs,
    ctx: RiscVContext,
    i: Inst,
    // &Utils,
    instrs: &[(u64, u32, BlockTarget)],
    input: InputRef<'_>,
    // memory: Memory,
    root: Func,
    // tune: &Tunables,
) {
    let memory = ctx.opts.mem;
    macro_rules! fallthrough {
        () => {{
            f.set_terminator(
                ctx.block,
                portal_pc_waffle::Terminator::Br {
                    target: BlockTarget {
                        block: ctx.rbs[4098],
                        args: r.to_args().into_iter().chain([ctx.pc_value]).collect(),
                    },
                },
            );
            return;
        }};
    }
    macro_rules! branch {
        ($e:expr) => {
            match $e {
                target => {
                    f.set_terminator(
                        ctx.block,
                        portal_pc_waffle::Terminator::Br {
                            target: BlockTarget {
                                block: f.entry,
                                args: r.to_args().into_iter().chain([target]).collect(),
                            },
                        },
                    );
                    return;
                }
            }
        };
    }
    macro_rules! wcond_branch {
        ($cond:expr => $e:expr) => {
            match $cond {
                cond => match $e {
                    target => {
                        f.set_terminator(
                            ctx.block,
                            portal_pc_waffle::Terminator::CondBr {
                                if_false: BlockTarget {
                                    block: f.entry,
                                    args: r.to_args().into_iter().chain([target]).collect(),
                                },
                                if_true: BlockTarget {
                                    block: ctx.rbs[4098],
                                    args: r.to_args().into_iter().chain([ctx.pc_value]).collect(),
                                },
                                cond: cond,
                            },
                        );
                        return;
                    }
                },
            }
        };
    }
    macro_rules! cond_branch {
        (if $c:expr => {$t:expr}else{$e:expr}) => {
            match $c {
                cond => match $t {
                    idx_true => match $e {
                        idx_false => {
                            let [idx_true,idx_false] = [idx_true,idx_false].map(|a|if a <= 0{
                                (4096 - a) as usize
                            }else{
                                a as usize
                            });
                            let [tpc, fpc] = [idx_true, idx_false].map(|a| {
                                let v = if a < 4096 {
                                    match instrs.get(ctx.local_instruction_index.wrapping_sub(a * 2)) {
                                        Some(a) => a.0,
                                        None => ctx.pc.wrapping_sub((a * 2).try_into().unwrap()),
                                    }
                                } else if a > 4096 {
                                    match instrs.get(ctx.local_instruction_index.wrapping_add((a - 4096) * 2)) {
                                        Some(a) => a.0,
                                        None => {
                                            ctx.pc.wrapping_add(((a - 4096) * 2).try_into().unwrap())
                                        }
                                    }
                                } else {
                                    return ctx.original_pc_value;
                                };
                                f.add_op(ctx.block, Operator::I64Const { value: v }, &[], &[Type::I64])
                            });
                            f.set_terminator(
                                ctx.block,
                                portal_pc_waffle::Terminator::CondBr {
                                    cond: cond,
                                    if_true: BlockTarget {
                                        block: ctx.rbs[idx_true],
                                        args: r.to_args().into_iter().chain([tpc]).collect(),
                                    },
                                    if_false: BlockTarget {
                                        block: ctx.rbs[idx_false],
                                        args: r.to_args().into_iter().chain([fpc]).collect(),
                                    },
                                },
                            );
                            return;
                        }
                    },
                },
            }
        };
    }
    macro_rules! arith {
        ($i:expr => {$([$r5:ident => $wasm:ident $({i $ii:ident})? $({W $w:ident})? $({iW $iiw:ident})?]),*} |$j:ident|$e:expr) => {
            match paste::paste!{match $i {
                $(
                    Inst::[<$r5>]{dest,src1,src2} => {
                        let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b));
                        let val = f.add_op(ctx.block,Operator::[<I64 $wasm>],&[src1,src2],&[if stringify!($r5).starts_with("slt"){
                            Type::I32
                        }else{
                            Type::I64
                        }]);
                        let val = if stringify!($r5).starts_with("Slt"){
                            f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64])
                        }else{
                            val
                        };
                        r.put(dest,val);
                        fallthrough!()
                    }
                    $(Inst::[<$r5 $w>]{dest,src1,src2} => {
                        let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b)).map(|b|{
                            f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        });
                        let val = f.add_op(ctx.block,Operator::[<I32 $wasm>],&[src1,src2],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    })?
                    $( Inst::[<$r5 $ii>]{dest,src1,imm} => {
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b));
                        let src2 = f.add_op(ctx.block,Operator::I64Const{value:imm.as_i64() as u64},&[],&[Type::I64]);
                          let val = f.add_op(ctx.block,Operator::[<I64 $wasm>],&[src1,src2],&[if stringify!($r5).starts_with("slt"){
                            Type::I32
                        }else{
                            Type::I64
                        }]);
                        let val = if stringify!($r5).starts_with("Slt"){
                            f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64])
                        }else{
                            val
                        };
                        r.put(dest,val);
                        fallthrough!()
                    }
                     )?
                     $(Inst::[<$r5 $iiw>]{dest,src1,imm} => {
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b)).map(|b|{
                            f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        });
                        let src2 = f.add_op(ctx.block,Operator::I32Const{value:imm.as_i32() as u32},&[],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::[<I32 $wasm>],&[src1,src2],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    })?
                ),*,
                a => Err(a),
            }}{
                Ok(a) => a,
                Err($j) => $e,
            }
        };
    }
    macro_rules! branches {
        ($i:expr => {$([$r5:ident => $wasm:ident]),*} |$j:ident|$e:expr) => {
            match paste::paste! {match $i {
                $(Inst::$r5{src1,src2,offset} => {
                     let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b));
                    let val = f.add_op(ctx.block,Operator::[<I64 $wasm>],&[src1,src2],&[Type::I32]);
                    cond_branch!(if val => {
                        offset.as_i32()
                    }else{
                        2
                    })
                }),*,
                a => Err(a)
            }} {
                Ok(a) => a,
                Err($j) => $e,
            }
        };
    }
    macro_rules! shifts {
        ($i:expr => {$([$r5:ident => $wasm:ident]),*} |$j:ident|$e:expr) => {
              match paste::paste!{match $i {
                $(
                    Inst::[<$r5>]{dest,src1,src2} => {
                        let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b));
                          let src2 = match src2{
                            b =>  f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        };
                        let val = f.add_op(ctx.block,Operator::[<I64 $wasm>],&[src1,src2],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    }
                    Inst::[<$r5 W>]{dest,src1,src2} => {
                        let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b)).map(|b|{
                            f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        });
                        let val = f.add_op(ctx.block,Operator::[<I32 $wasm>],&[src1,src2],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    }
                     Inst::[<$r5 i>]{dest,src1,imm} => {
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b));
                        let src2 = f.add_op(ctx.block,Operator::I64Const{value:imm.as_i64() as u64},&[],&[Type::I64]);
                        let src2 = match src2{
                            b =>  f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        };
                        let val = f.add_op(ctx.block,Operator::[<I64 $wasm>],&[src1,src2],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    }
                     Inst::[<$r5 iW>]{dest,src1,imm} => {
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b)).map(|b|{
                            f.add_op(ctx.block,Operator::I32WrapI64,&[b],&[Type::I32])
                        });
                        let src2 = f.add_op(ctx.block,Operator::I32Const{value:imm.as_i32() as u32},&[],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::[<I32 $wasm>],&[src1,src2],&[Type::I32]);
                        let val = f.add_op(ctx.block,Operator::I64ExtendI32U,&[val],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    }
                ),* ,
                a => Err(a),
            }}{
                Ok(a) => a,
                Err($j) => $e,
            }
        };
    }
    macro_rules! loads {
        ($i:expr => {$([$r5:ident => $v:literal $wasm:ident ]),*} |$j:ident|$e:expr) => {
            match paste::paste! {match $i {
                $(Inst::$r5{base,dest,offset} => match $v{v => {
                    let src1 = base;
                    let [src1] = [src1].map(|b|r.get(f,ctx.block,b));
                    let offset = f.add_op(ctx.block,Operator::I64Const{value: offset.as_i64() as u64},&[],&[Type::I64]);
                    let src1 = f.add_op(ctx.block,Operator::I64Add,&[src1,offset],&[Type::I64]);
                    let src1 = ctx.opts.map(module,f, ctx.block, src1, &mut r.user);
                    let val = f.add_op(ctx.block,Operator::[<I64Load $v $wasm>]{memory:MemoryArg{offset:0,align:u32::trailing_zeros(v)-3,memory}},&[src1],&[Type::I64]);
                       r.put(dest,val);
                        fallthrough!()
                }}),*,
                a => Err(a)
            }} {
                Ok(a) => a,
                Err($j) => $e,
            }
        };
    }
    macro_rules! stores {
        ($i:expr => {$([$r5:ident => $v:literal  ]),*} |$j:ident|$e:expr) => {
            match paste::paste! {match $i {
                $(Inst::$r5{base,src,offset} => match $v{v => {
                    let src1 = base;
                    let src2 = src;
                    let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b));
                    let offset = f.add_op(ctx.block,Operator::I64Const{value: offset.as_i64() as u64},&[],&[Type::I64]);
                    let src1 = f.add_op(ctx.block,Operator::I64Add,&[src1,offset],&[Type::I64]);
                    let src1 = ctx.opts.map(module,f, ctx.block, src1, &mut r.user);
                    f.add_op(ctx.block,Operator::[<I64Store $v>]{memory:MemoryArg{offset:0,align:u32::trailing_zeros(v)-3,memory}},&[src1,src2],&[Type::I64]);
                    //    r.put(dest,val);
                        fallthrough!()
                }}),*,
                a => Err(a)
            }} {
                Ok(a) => a,
                Err($j) => $e,
            }
        };
    }
    arith!(i => {[Add => Add { i i} {W W} {iW iW}],[Sub => Sub{W W}],[Mul => Mul{W W}],[Div => DivS{W W}],[Divu => DivU{W W}],[And => And {i i}],[Or => Or {i i}],[Xor => Xor {i i}],[Slt => LtS { i i}],[Sltu => LtU ]} | i|match i{
        Inst::Sltiu{dest,src1,imm} => {
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b));
                        let src2 = f.add_op(ctx.block,Operator::I64Const{value:imm.as_i64() as u64},&[],&[Type::I64]);
                        let val = f.add_op(ctx.block,Operator::I64LtU,&[src1,src2],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    },
         i => shifts!(i => {[Sll => Shl],[Srl => ShrU],[Sra => ShrS]} |i|match i{
            Inst::Lui { uimm, dest } => {
                 let val = f.add_op(ctx.block,Operator::I64Const { value: uimm.as_u64() },&[],&[Type::I64]);
                        r.put(dest,val);
                fallthrough!()
            }
             Inst::Auipc { uimm, dest } => {
                 let val = f.add_op(ctx.block,Operator::I64Const { value: uimm.as_u64().wrapping_add(ctx.pc) },&[],&[Type::I64]);
                        r.put(dest,val);
                fallthrough!()
            }
            Inst::Jal { offset, dest } => {
                r.put(dest,ctx.pc_value);
                let val = f.add_op(ctx.block,Operator::I64Const { value: offset.as_u64().wrapping_add(ctx.pc) },&[],&[Type::I64]);
                branch!(val)
            }
            Inst::Jalr { offset, base, dest } => {
                let b = r.get(f, ctx.block, base);
                 r.put(dest,ctx.pc_value);
                let val = f.add_op(ctx.block,Operator::I64Const { value: offset.as_u64() },&[],&[Type::I64]);
                let val = f.add_op(ctx.block,Operator::I64Add,&[val,b],&[Type::I64]);
                let one = f.add_op(ctx.block, Operator::I64Const { value: 1 }, &[], &[Type::I64]);
                let val = [Operator::I64Or,Operator::I64Sub].into_iter().fold(val, |v,o|{
                    f.add_op(ctx.block,o,&[v,one],&[Type::I64])
                });
                branch!(val)
            }
            i => branches!(i => {[Beq => Eq],[Bne => Ne],[Blt => LtS],[Bltu => LtU],[Bge => GeS],[Bgeu => GeU]} |i|match i{
                i => loads!(i => {[Lb => 8 S], [Lbu => 8 U], [Lh => 16 S],[Lhu => 16 U],[Lw => 32 S],[Lwu => 32 U]} |i|match i{
                    Inst::Ld{base,dest,offset} => {
                        let src1 = base;
                        let [src1] = [src1].map(|b|r.get(f,ctx.block,b));
                        let offset = f.add_op(ctx.block,Operator::I64Const{value: offset.as_i64() as u64},&[],&[Type::I64]);
                        let src1 = f.add_op(ctx.block,Operator::I64Add,&[src1,offset],&[Type::I64]);
                        let src1 = ctx.opts.map(module,f, ctx.block, src1, &mut r.user);
                        let val = f.add_op(ctx.block,Operator::I64Load{memory:MemoryArg{offset:0,align:3,memory}},&[src1],&[Type::I64]);
                        r.put(dest,val);
                        fallthrough!()
                    }
                    i => stores!(i => {[Sb => 8],[Sh => 16],[Sw => 32]} |i|match i{
                        Inst::Sd{base,src,offset} => {
                            let src1 = base;
                            let src2 = src;
                            let [src1,src2] = [src1,src2].map(|b|r.get(f,ctx.block,b));
                            let offset = f.add_op(ctx.block,Operator::I64Const{value: offset.as_i64() as u64},&[],&[Type::I64]);
                            let src1 = f.add_op(ctx.block,Operator::I64Add,&[src1,offset],&[Type::I64]);
                            let src1 = ctx.opts.map(module,f, ctx.block, src1, &mut r.user);
                            f.add_op(ctx.block,Operator::I64Store{memory:MemoryArg{offset:0,align:3,memory}},&[src1,src2],&[Type::I64]);
                            //    r.put(dest,val);
                            fallthrough!()
                        },
                        Inst::Fence { fence: _ } => {
                            fallthrough!()
                        },
                        Inst::Ecall => {
                            if ctx.opts.inline_ecall{
                                let SignatureData::Func { params, returns, shared } = &module.signatures[module.funcs[ctx.opts.ecall].sig()] else{
                                    unreachable!()
                                };
                                let x = f.add_op(ctx.block,Operator::Call { function_index: ctx.opts.ecall },&r.to_args().chain([ctx.pc_value]).collect_vec(),&returns);
                                let mut x = results_ref_2(f, x).into_iter();
                                *r = Regs::from_args(&r.user, &mut x).unwrap();
                                match x.next(){
                                    None => {
                                        fallthrough!()
                                    },
                                    Some(v) => match x.next(){
                                        Some(w) => {
                                        // let w = f.add_op(ctx.block, Operator::I64Eqz, &[v], &[Type::I32]);
                                            wcond_branch!(w => v);
                                        }, None => {
                                            branch!(v);
                                        }
                                    }
                                }
                            }else{
                                let stash = 'a: {
                                    for (i,v) in module.tables[ctx.opts.table].func_elements.as_mut().unwrap().iter_mut().enumerate(){
                                        if *v == root{
                                            break 'a i;
                                        }
                                    }
                                    let t = module.tables[ctx.opts.table].func_elements.as_mut().unwrap();
                                    let i = t.len();
                                    t.push(root);
                                    break 'a i;
                                };
                                let stash = stash as u64;
                                let stash = f.add_op(ctx.block, Operator::I64Const { value: stash }, &[], &[Type::I64]);
                                f.set_terminator(ctx.block, portal_pc_waffle::Terminator::ReturnCall { func: ctx.opts.ecall, args: [stash].into_iter().chain(r.to_args()).chain([ctx.pc_value]).collect() });
                            }
                        },
                        
                        // =========================================================================
                        // F Extension: Single-Precision Floating-Point Instructions
                        // =========================================================================
                        // RISC-V Specification Quote:
                        // "This chapter describes the standard instruction-set extension for
                        // single-precision floating-point, which is named 'F' and adds
                        // single-precision floating-point computational instructions compliant
                        // with the IEEE 754-2008 arithmetic standard."
                        
                        // FLW: Load Floating-Point Word
                        // RISC-V Specification Quote:
                        // "Floating-point loads and stores use the same base+offset addressing
                        // mode as the integer base ISA, with a base address in register rs1
                        // and a 12-bit signed byte offset."
                        Inst::Flw { offset, dest, base } => {
                            let base_val = r.get(f, ctx.block, base);
                            let offset_val = f.add_op(ctx.block, Operator::I64Const { value: offset.as_i64() as u64 }, &[], &[Type::I64]);
                            let addr = f.add_op(ctx.block, Operator::I64Add, &[base_val, offset_val], &[Type::I64]);
                            let addr = ctx.opts.map(module, f, ctx.block, addr, &mut r.user);
                            // Load 32-bit float, then promote to 64-bit for storage
                            // RISC-V Specification Quote:
                            // "When a single-precision floating-point value is loaded into a
                            // floating-point register that supports wider formats, the value
                            // is NaN-boxed according to the NaN-boxing rules."
                            let val = f.add_op(ctx.block, Operator::F32Load { memory: MemoryArg { offset: 0, align: 2, memory } }, &[addr], &[Type::F32]);
                            let val = nan_box_f32(f, ctx.block, val);
                            r.put_f(dest, val);
                            fallthrough!()
                        },
                        
                        // FSW: Store Floating-Point Word
                        // RISC-V Specification Quote:
                        // "FSW stores a single-precision value from floating-point register rs2 to memory."
                        Inst::Fsw { offset, src, base } => {
                            let base_val = r.get(f, ctx.block, base);
                            let offset_val = f.add_op(ctx.block, Operator::I64Const { value: offset.as_i64() as u64 }, &[], &[Type::I64]);
                            let addr = f.add_op(ctx.block, Operator::I64Add, &[base_val, offset_val], &[Type::I64]);
                            let addr = ctx.opts.map(module, f, ctx.block, addr, &mut r.user);
                            // Demote 64-bit value to 32-bit for store
                            let val = r.get_f(src);
                            let val = nan_unbox_to_f32(f, ctx.block, val);
                            f.add_op(ctx.block, Operator::F32Store { memory: MemoryArg { offset: 0, align: 2, memory } }, &[addr, val], &[]);
                            fallthrough!()
                        },
                        
                        // FADD.S: Add Single-Precision
                        // RISC-V Specification Quote:
                        // "Floating-point arithmetic instructions with one or two source operands
                        // use the R-type format with the OP-FP major opcode."
                        Inst::FaddS { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            // Demote to F32, compute, promote back
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Add, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSUB.S: Subtract Single-Precision
                        Inst::FsubS { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Sub, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMUL.S: Multiply Single-Precision
                        Inst::FmulS { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Mul, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FDIV.S: Divide Single-Precision
                        Inst::FdivS { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Div, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSQRT.S: Square Root Single-Precision
                        // RISC-V Specification Quote:
                        // "Floating-point square root is encoded with rs2=0."
                        Inst::FsqrtS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::F32Sqrt, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMIN.S: Minimum Single-Precision
                        // RISC-V Specification Quote:
                        // "For FMIN.S and FMAX.S, if at least one input is a signaling NaN,
                        // or if both inputs are quiet NaNs, the result is the canonical NaN.
                        // If one operand is a quiet NaN and the other is not a NaN, the
                        // result is the non-NaN operand."
                        Inst::FminS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Min, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMAX.S: Maximum Single-Precision
                        Inst::FmaxS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Max, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJ.S: Sign-Inject Single-Precision (copy sign from src2 to src1)
                        // RISC-V Specification Quote:
                        // "FSGNJ.S, FSGNJN.S, and FSGNJX.S produce a result that takes all
                        // bits except the sign bit from rs1. For FSGNJ, the result's sign
                        // bit is rs2's sign bit."
                        Inst::FsgnjS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let result = f.add_op(ctx.block, Operator::F32Copysign, &[s1, s2], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJN.S: Sign-Inject-Negate Single-Precision
                        // RISC-V Specification Quote:
                        // "For FSGNJN, the result's sign bit is the opposite of rs2's sign bit."
                        Inst::FsgnjnS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            // Negate src2 to flip its sign, then copysign
                            let s2_neg = f.add_op(ctx.block, Operator::F32Neg, &[s2], &[Type::F32]);
                            let result = f.add_op(ctx.block, Operator::F32Copysign, &[s1, s2_neg], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJX.S: Sign-Inject-XOR Single-Precision
                        // RISC-V Specification Quote:
                        // "For FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2."
                        Inst::FsgnjxS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            // XOR sign bits via reinterpret + XOR + reinterpret
                            let i1 = f.add_op(ctx.block, Operator::I64ReinterpretF64, &[s1], &[Type::I64]);
                            let i2 = f.add_op(ctx.block, Operator::I64ReinterpretF64, &[s2], &[Type::I64]);
                            // Sign bit is at bit 63
                            let sign_mask = f.add_op(ctx.block, Operator::I64Const { value: 0x8000_0000_0000_0000 }, &[], &[Type::I64]);
                            let i2_sign = f.add_op(ctx.block, Operator::I64And, &[i2, sign_mask], &[Type::I64]);
                            let i1_xor = f.add_op(ctx.block, Operator::I64Xor, &[i1, i2_sign], &[Type::I64]);
                            let result = f.add_op(ctx.block, Operator::F64ReinterpretI64, &[i1_xor], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FEQ.S: Floating-Point Equal Single-Precision
                        // RISC-V Specification Quote:
                        // "FEQ.S performs a quiet comparison: it only sets the invalid operation
                        // exception flag if either input is a signaling NaN."
                        Inst::FeqS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let cmp = f.add_op(ctx.block, Operator::F32Eq, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FLT.S: Floating-Point Less Than Single-Precision
                        // RISC-V Specification Quote:
                        // "FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to
                        // as signaling comparisons: that is, they set the invalid operation
                        // exception flag if either input is NaN."
                        Inst::FltS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let cmp = f.add_op(ctx.block, Operator::F32Lt, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FLE.S: Floating-Point Less Than or Equal Single-Precision
                        Inst::FleS { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let cmp = f.add_op(ctx.block, Operator::F32Le, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.W.S: Convert Single to Signed Word
                        // RISC-V Specification Quote:
                        // "Floating-point-to-integer and integer-to-floating-point conversion
                        // instructions are encoded in the OP-FP major opcode space."
                        Inst::FcvtWS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::I32TruncF32S, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32S, &[result], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.WU.S: Convert Single to Unsigned Word
                        Inst::FcvtWuS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::I32TruncF32U, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[result], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.S.W: Convert Signed Word to Single
                        Inst::FcvtSW { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let s = f.add_op(ctx.block, Operator::I32WrapI64, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::F32ConvertI32S, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.S.WU: Convert Unsigned Word to Single
                        Inst::FcvtSWu { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let s = f.add_op(ctx.block, Operator::I32WrapI64, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::F32ConvertI32U, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMV.X.W: Move from Floating-Point to Integer Register
                        // RISC-V Specification Quote:
                        // "FMV.X.W instruction moves the single-precision value in floating-point
                        // register rs1 represented in IEEE 754-2008 encoding to the lower 32 bits
                        // of integer register rd."
                        Inst::FmvXW { dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::I32ReinterpretF32, &[s], &[Type::I32]);
                            // Sign-extend to 64 bits (per RV64F spec)
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32S, &[result], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FMV.W.X: Move from Integer to Floating-Point Register
                        // RISC-V Specification Quote:
                        // "FMV.W.X instruction moves the single-precision value encoded in
                        // IEEE 754-2008 standard encoding from the lower 32 bits of integer
                        // register rs1 to the floating-point register rd."
                        Inst::FmvWX { dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let s = f.add_op(ctx.block, Operator::I32WrapI64, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::F32ReinterpretI32, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCLASS.S: Floating-Point Classify Single-Precision
                        // RISC-V Specification Quote:
                        // "The FCLASS.S instruction examines the value in floating-point register
                        // rs1 and writes to integer register rd a 10-bit mask that indicates the
                        // class of the floating-point number."
                        // Returns a 10-bit mask (bit 0-9) indicating class
                        Inst::FclassS { dest, src: _ } => {
                            // FCLASS.S is complex to implement properly in WebAssembly as it requires
                            // distinguishing between quiet NaN, signaling NaN, infinity, zero, 
                            // subnormal, and normal numbers with both positive and negative variants.
                            // For now, return a placeholder value (bit 8 = positive normal).
                            // TODO: Full implementation requires checking NaN, Inf, zero, subnormal
                            let result = f.add_op(ctx.block, Operator::I64Const { value: 0x100 }, &[], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // Fused Multiply-Add Instructions
                        // RISC-V Specification Quote:
                        // "The fused multiply-add instructions must set the invalid operation
                        // exception flag when the multiplicands are ∞ and zero, even when the
                        // addend is a quiet NaN."
                        
                        // FMADD.S: (src1 * src2) + src3
                        Inst::FmaddS { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let s3 = nan_unbox_to_f32(f, ctx.block, s3);
                            // Note: WebAssembly doesn't have fused multiply-add, so we use separate
                            // multiply and add operations. This may produce slightly different results
                            // than a true FMA due to intermediate rounding (one rounding vs none).
                            // RISC-V Specification Quote:
                            // "The fused multiply-add instructions are defined to compute (rs1×rs2)+rs3
                            // with a single rounding."
                            let mul = f.add_op(ctx.block, Operator::F32Mul, &[s1, s2], &[Type::F32]);
                            let result = f.add_op(ctx.block, Operator::F32Add, &[mul, s3], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMSUB.S: (src1 * src2) - src3
                        Inst::FmsubS { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let s3 = nan_unbox_to_f32(f, ctx.block, s3);
                            let mul = f.add_op(ctx.block, Operator::F32Mul, &[s1, s2], &[Type::F32]);
                            let result = f.add_op(ctx.block, Operator::F32Sub, &[mul, s3], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FNMSUB.S: -(src1 * src2) + src3
                        Inst::FnmsubS { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let s3 = nan_unbox_to_f32(f, ctx.block, s3);
                            let mul = f.add_op(ctx.block, Operator::F32Mul, &[s1, s2], &[Type::F32]);
                            let neg_mul = f.add_op(ctx.block, Operator::F32Neg, &[mul], &[Type::F32]);
                            let result = f.add_op(ctx.block, Operator::F32Add, &[neg_mul, s3], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FNMADD.S: -(src1 * src2) - src3
                        Inst::FnmaddS { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let s1 = nan_unbox_to_f32(f, ctx.block, s1);
                            let s2 = nan_unbox_to_f32(f, ctx.block, s2);
                            let s3 = nan_unbox_to_f32(f, ctx.block, s3);
                            let mul = f.add_op(ctx.block, Operator::F32Mul, &[s1, s2], &[Type::F32]);
                            let neg_mul = f.add_op(ctx.block, Operator::F32Neg, &[mul], &[Type::F32]);
                            let result = f.add_op(ctx.block, Operator::F32Sub, &[neg_mul, s3], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // =========================================================================
                        // D Extension: Double-Precision Floating-Point Instructions
                        // =========================================================================
                        // RISC-V Specification Quote:
                        // "This chapter describes the standard double-precision floating-point
                        // instruction-set extension, which is named 'D' and adds double-precision
                        // floating-point computational instructions compliant with the
                        // IEEE 754-2008 arithmetic standard."
                        
                        // FLD: Load Floating-Point Double
                        // RISC-V Specification Quote:
                        // "The FLD instruction loads a double-precision floating-point value
                        // from memory into floating-point register rd."
                        Inst::Fld { offset, dest, base } => {
                            let base_val = r.get(f, ctx.block, base);
                            let offset_val = f.add_op(ctx.block, Operator::I64Const { value: offset.as_i64() as u64 }, &[], &[Type::I64]);
                            let addr = f.add_op(ctx.block, Operator::I64Add, &[base_val, offset_val], &[Type::I64]);
                            let addr = ctx.opts.map(module, f, ctx.block, addr, &mut r.user);
                            let val = f.add_op(ctx.block, Operator::F64Load { memory: MemoryArg { offset: 0, align: 3, memory } }, &[addr], &[Type::F64]);
                            r.put_f(dest, val);
                            fallthrough!()
                        },
                        
                        // FSD: Store Floating-Point Double
                        // RISC-V Specification Quote:
                        // "The FSD instruction stores a double-precision value from the
                        // floating-point registers to memory."
                        Inst::Fsd { offset, src, base } => {
                            let base_val = r.get(f, ctx.block, base);
                            let offset_val = f.add_op(ctx.block, Operator::I64Const { value: offset.as_i64() as u64 }, &[], &[Type::I64]);
                            let addr = f.add_op(ctx.block, Operator::I64Add, &[base_val, offset_val], &[Type::I64]);
                            let addr = ctx.opts.map(module, f, ctx.block, addr, &mut r.user);
                            let val = r.get_f(src);
                            f.add_op(ctx.block, Operator::F64Store { memory: MemoryArg { offset: 0, align: 3, memory } }, &[addr, val], &[]);
                            fallthrough!()
                        },
                        
                        // FADD.D: Add Double-Precision
                        Inst::FaddD { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Add, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSUB.D: Subtract Double-Precision
                        Inst::FsubD { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Sub, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMUL.D: Multiply Double-Precision
                        Inst::FmulD { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Mul, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FDIV.D: Divide Double-Precision
                        Inst::FdivD { rm: _, dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Div, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSQRT.D: Square Root Double-Precision
                        Inst::FsqrtD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::F64Sqrt, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMIN.D: Minimum Double-Precision
                        Inst::FminD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Min, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMAX.D: Maximum Double-Precision
                        Inst::FmaxD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Max, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJ.D: Sign-Inject Double-Precision
                        Inst::FsgnjD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let result = f.add_op(ctx.block, Operator::F64Copysign, &[s1, s2], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJN.D: Sign-Inject-Negate Double-Precision
                        Inst::FsgnjnD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s2_neg = f.add_op(ctx.block, Operator::F64Neg, &[s2], &[Type::F64]);
                            let result = f.add_op(ctx.block, Operator::F64Copysign, &[s1, s2_neg], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FSGNJX.D: Sign-Inject-XOR Double-Precision
                        Inst::FsgnjxD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let i1 = f.add_op(ctx.block, Operator::I64ReinterpretF64, &[s1], &[Type::I64]);
                            let i2 = f.add_op(ctx.block, Operator::I64ReinterpretF64, &[s2], &[Type::I64]);
                            let sign_mask = f.add_op(ctx.block, Operator::I64Const { value: 0x8000_0000_0000_0000 }, &[], &[Type::I64]);
                            let i2_sign = f.add_op(ctx.block, Operator::I64And, &[i2, sign_mask], &[Type::I64]);
                            let i1_xor = f.add_op(ctx.block, Operator::I64Xor, &[i1, i2_sign], &[Type::I64]);
                            let result = f.add_op(ctx.block, Operator::F64ReinterpretI64, &[i1_xor], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FEQ.D: Floating-Point Equal Double-Precision
                        Inst::FeqD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let cmp = f.add_op(ctx.block, Operator::F64Eq, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FLT.D: Floating-Point Less Than Double-Precision
                        Inst::FltD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let cmp = f.add_op(ctx.block, Operator::F64Lt, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FLE.D: Floating-Point Less Than or Equal Double-Precision
                        Inst::FleD { dest, src1, src2 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let cmp = f.add_op(ctx.block, Operator::F64Le, &[s1, s2], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[cmp], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.S.D: Convert Double to Single
                        // RISC-V Specification Quote:
                        // "FCVT.S.D rounds a double-precision floating-point number to
                        // single precision."
                        Inst::FcvtSD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            // Convert double to single using F32DemoteF64, then NaN-box the result
                            let result = f.add_op(ctx.block, Operator::F32DemoteF64, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.D.S: Convert Single to Double
                        // RISC-V Specification Quote:
                        // "FCVT.D.S extends a single-precision floating-point number to
                        // double precision."
                        Inst::FcvtDS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            // Unbox the single-precision value, then promote to double
                            let s32 = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::F64PromoteF32, &[s32], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.W.D: Convert Double to Signed Word
                        Inst::FcvtWD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::I32TruncF64S, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32S, &[result], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.WU.D: Convert Double to Unsigned Word
                        Inst::FcvtWuD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::I32TruncF64U, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::I64ExtendI32U, &[result], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.D.W: Convert Signed Word to Double
                        Inst::FcvtDW { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let s = f.add_op(ctx.block, Operator::I32WrapI64, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::F64ConvertI32S, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.D.WU: Convert Unsigned Word to Double
                        Inst::FcvtDWu { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let s = f.add_op(ctx.block, Operator::I32WrapI64, &[s], &[Type::I32]);
                            let result = f.add_op(ctx.block, Operator::F64ConvertI32U, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCLASS.D: Floating-Point Classify Double-Precision
                        Inst::FclassD { dest, src: _ } => {
                            // FCLASS.D is complex to implement properly in WebAssembly - see FCLASS.S comment.
                            // TODO: Full implementation requires checking NaN, Inf, zero, subnormal
                            let result = f.add_op(ctx.block, Operator::I64Const { value: 0x100 }, &[], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // Double-Precision Fused Multiply-Add Instructions
                        
                        // FMADD.D: (src1 * src2) + src3
                        Inst::FmaddD { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let mul = f.add_op(ctx.block, Operator::F64Mul, &[s1, s2], &[Type::F64]);
                            let result = f.add_op(ctx.block, Operator::F64Add, &[mul, s3], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMSUB.D: (src1 * src2) - src3
                        Inst::FmsubD { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let mul = f.add_op(ctx.block, Operator::F64Mul, &[s1, s2], &[Type::F64]);
                            let result = f.add_op(ctx.block, Operator::F64Sub, &[mul, s3], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FNMSUB.D: -(src1 * src2) + src3
                        Inst::FnmsubD { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let mul = f.add_op(ctx.block, Operator::F64Mul, &[s1, s2], &[Type::F64]);
                            let neg_mul = f.add_op(ctx.block, Operator::F64Neg, &[mul], &[Type::F64]);
                            let result = f.add_op(ctx.block, Operator::F64Add, &[neg_mul, s3], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FNMADD.D: -(src1 * src2) - src3
                        Inst::FnmaddD { rm: _, dest, src1, src2, src3 } => {
                            let s1 = r.get_f(src1);
                            let s2 = r.get_f(src2);
                            let s3 = r.get_f(src3);
                            let mul = f.add_op(ctx.block, Operator::F64Mul, &[s1, s2], &[Type::F64]);
                            let neg_mul = f.add_op(ctx.block, Operator::F64Neg, &[mul], &[Type::F64]);
                            let result = f.add_op(ctx.block, Operator::F64Sub, &[neg_mul, s3], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // =========================================================================
                        // RV64F/D: 64-bit Floating-Point Instructions
                        // =========================================================================
                        // RISC-V Specification Quote:
                        // "FCVT.L[U].S, FCVT.S.L[U], FCVT.L[U].D, and FCVT.D.L[U] variants
                        // convert to or from a signed or unsigned 64-bit integer, respectively."
                        
                        // FCVT.L.S: Convert Single to Signed Long (RV64F)
                        Inst::FcvtLS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::I64TruncF32S, &[s], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.LU.S: Convert Single to Unsigned Long (RV64F)
                        Inst::FcvtLuS { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let s = nan_unbox_to_f32(f, ctx.block, s);
                            let result = f.add_op(ctx.block, Operator::I64TruncF32U, &[s], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.S.L: Convert Signed Long to Single (RV64F)
                        Inst::FcvtSL { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let result = f.add_op(ctx.block, Operator::F32ConvertI64S, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.S.LU: Convert Unsigned Long to Single (RV64F)
                        Inst::FcvtSLu { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let result = f.add_op(ctx.block, Operator::F32ConvertI64U, &[s], &[Type::F32]);
                            let result = nan_box_f32(f, ctx.block, result);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.L.D: Convert Double to Signed Long (RV64D)
                        Inst::FcvtLD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::I64TruncF64S, &[s], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.LU.D: Convert Double to Unsigned Long (RV64D)
                        Inst::FcvtLuD { rm: _, dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::I64TruncF64U, &[s], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.D.L: Convert Signed Long to Double (RV64D)
                        Inst::FcvtDL { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let result = f.add_op(ctx.block, Operator::F64ConvertI64S, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FCVT.D.LU: Convert Unsigned Long to Double (RV64D)
                        Inst::FcvtDLu { rm: _, dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let result = f.add_op(ctx.block, Operator::F64ConvertI64U, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // FMV.X.D: Move Double to Integer Register (RV64D)
                        // RISC-V Specification Quote:
                        // "FMV.X.D moves the double-precision value in floating-point register
                        // rs1 to a representation in IEEE 754-2008 standard encoding in integer
                        // register rd."
                        Inst::FmvXD { dest, src } => {
                            let s = r.get_f(src);
                            let result = f.add_op(ctx.block, Operator::I64ReinterpretF64, &[s], &[Type::I64]);
                            r.put(dest, result);
                            fallthrough!()
                        },
                        
                        // FMV.D.X: Move Integer to Double Register (RV64D)
                        // RISC-V Specification Quote:
                        // "FMV.D.X moves the double-precision value encoded in IEEE 754-2008
                        // standard encoding from integer register rs1 to the floating-point
                        // register rd."
                        Inst::FmvDX { dest, src } => {
                            let s = r.get(f, ctx.block, src);
                            let result = f.add_op(ctx.block, Operator::F64ReinterpretI64, &[s], &[Type::F64]);
                            r.put_f(dest, result);
                            fallthrough!()
                        },
                        
                        // a => todo!("unhandled op: {a:?}")
                        a => {
                            let rv = f.add_op(ctx.block, Operator::I32Const { value: 1 }, &[], &[Type::I32]);
                            f.set_terminator(
                                ctx.block,
                                portal_pc_waffle::Terminator::Return {
                                    values: r.to_args().chain([ctx.pc_value]).chain(once(rv)).collect(),
                                },
                            );
              
                        }
                    }),
                })
            })
         }),
    });
    // fallthrough!()
}

/// Helper type for page table base address that can be either a runtime Value or a static constant
pub enum PageTableBase {
    /// Runtime value (e.g., from a local, param, or global)
    Runtime(Value),
    /// Static constant address
    Constant(u64),
}

impl From<Value> for PageTableBase {
    fn from(v: Value) -> Self {
        PageTableBase::Runtime(v)
    }
}

impl From<u64> for PageTableBase {
    fn from(c: u64) -> Self {
        PageTableBase::Constant(c)
    }
}

impl PageTableBase {
    fn to_value(self, f: &mut FunctionBody, block: Block) -> Value {
        match self {
            PageTableBase::Runtime(v) => v,
            PageTableBase::Constant(c) => f.add_op(block, Operator::I64Const { value: c }, &[], &[Type::I64]),
        }
    }
}

/// Standard page table mapper for 64KB single-level paging
///
/// This helper generates WebAssembly code to translate virtual addresses using a flat page table.
/// The page table base address can be provided as either a runtime Value or a static constant.
///
/// # Page Table Format
/// - Each entry is 8 bytes (i64) containing the physical page base address
/// - Entry address = page_table_base + (page_num * 8)
/// - Page number = vaddr >> 16 (bits 63:16)
/// - Page offset = vaddr & 0xFFFF (bits 15:0)
///
/// # Arguments
/// - `module`: WebAssembly module
/// - `f`: Function body being built
/// - `block`: Current block
/// - `vaddr`: Virtual address value
/// - `page_table_base`: Base address of page table (runtime Value or static u64)
/// - `memory`: Memory index to use for loads
///
/// # Returns
/// Physical address value
pub fn standard_page_table_mapper(
    module: &mut Module,
    f: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    page_table_base: impl Into<PageTableBase>,
    memory: Memory,
) -> Value {
    let pt_base_value = page_table_base.into().to_value(f, block);
    
    // Extract page number: vaddr >> 16
    let shift_16 = f.add_op(block, Operator::I64Const { value: 16 }, &[], &[Type::I64]);
    let page_num = f.add_op(block, Operator::I64ShrU, &[vaddr, shift_16], &[Type::I64]);
    
    // Multiply page_num by 8 (size of u64 entry)
    let shift_3 = f.add_op(block, Operator::I64Const { value: 3 }, &[], &[Type::I64]);
    let entry_offset = f.add_op(block, Operator::I64Shl, &[page_num, shift_3], &[Type::I64]);
    
    // Add page table base address
    let entry_addr = f.add_op(block, Operator::I64Add, &[pt_base_value, entry_offset], &[Type::I64]);
    
    // Load physical page base from page table
    let phys_page = f.add_op(
        block,
        Operator::I64Load {
            memory: MemoryArg {
                offset: 0,
                align: 3,
                memory,
            }
        },
        &[entry_addr],
        &[Type::I64]
    );
    
    // Extract page offset: vaddr & 0xFFFF
    let mask = f.add_op(block, Operator::I64Const { value: 0xFFFF }, &[], &[Type::I64]);
    let page_offset = f.add_op(block, Operator::I64And, &[vaddr, mask], &[Type::I64]);
    
    // Combine: phys_page + page_offset
    f.add_op(block, Operator::I64Add, &[phys_page, page_offset], &[Type::I64])
}

/// Multi-level page table mapper for 64KB pages
///
/// This helper generates WebAssembly code for a 3-level page table structure.
/// Each level uses 16-bit indices, supporting the full 64-bit address space.
///
/// # Page Table Structure
/// - Level 3 (top): Indexed by bits [63:48]
/// - Level 2: Indexed by bits [47:32]
/// - Level 1 (leaf): Indexed by bits [31:16], contains physical page bases
/// - Page offset: bits [15:0]
///
/// # Arguments
/// - `module`: WebAssembly module
/// - `f`: Function body being built
/// - `block`: Current block
/// - `vaddr`: Virtual address value
/// - `l3_table_base`: Base address of level 3 page table (runtime Value or static u64)
/// - `memory`: Memory index to use for loads
///
/// # Returns
/// Physical address value
pub fn multilevel_page_table_mapper(
    module: &mut Module,
    f: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    l3_table_base: impl Into<PageTableBase>,
    memory: Memory,
) -> Value {
    let l3_base_value = l3_table_base.into().to_value(f, block);
    // Helper to extract a 16-bit field from vaddr
    let extract_16bit = |f: &mut FunctionBody, block, val: Value, shift_amt: u64| -> Value {
        let shift = f.add_op(block, Operator::I64Const { value: shift_amt }, &[], &[Type::I64]);
        let shifted = f.add_op(block, Operator::I64ShrU, &[val, shift], &[Type::I64]);
        let mask = f.add_op(block, Operator::I64Const { value: 0xFFFF }, &[], &[Type::I64]);
        f.add_op(block, Operator::I64And, &[shifted, mask], &[Type::I64])
    };
    
    // Level 3: bits [63:48]
    let l3_idx = extract_16bit(f, block, vaddr, 48);
    let shift_3 = f.add_op(block, Operator::I64Const { value: 3 }, &[], &[Type::I64]);
    let l3_offset = f.add_op(block, Operator::I64Shl, &[l3_idx, shift_3], &[Type::I64]);
    let l3_entry_addr = f.add_op(block, Operator::I64Add, &[l3_base_value, l3_offset], &[Type::I64]);
    let l2_table_base = f.add_op(
        block,
        Operator::I64Load { memory: MemoryArg { offset: 0, align: 3, memory } },
        &[l3_entry_addr],
        &[Type::I64]
    );
    
    // Level 2: bits [47:32]
    let l2_idx = extract_16bit(f, block, vaddr, 32);
    let l2_offset = f.add_op(block, Operator::I64Shl, &[l2_idx, shift_3], &[Type::I64]);
    let l2_entry_addr = f.add_op(block, Operator::I64Add, &[l2_table_base, l2_offset], &[Type::I64]);
    let l1_table_base = f.add_op(
        block,
        Operator::I64Load { memory: MemoryArg { offset: 0, align: 3, memory } },
        &[l2_entry_addr],
        &[Type::I64]
    );
    
    // Level 1: bits [31:16]
    let l1_idx = extract_16bit(f, block, vaddr, 16);
    let l1_offset = f.add_op(block, Operator::I64Shl, &[l1_idx, shift_3], &[Type::I64]);
    let l1_entry_addr = f.add_op(block, Operator::I64Add, &[l1_table_base, l1_offset], &[Type::I64]);
    let phys_page = f.add_op(
        block,
        Operator::I64Load { memory: MemoryArg { offset: 0, align: 3, memory } },
        &[l1_entry_addr],
        &[Type::I64]
    );
    
    // Page offset: bits [15:0]
    let page_offset = extract_16bit(f, block, vaddr, 0);
    
    // Combine: phys_page + page_offset
    f.add_op(block, Operator::I64Add, &[phys_page, page_offset], &[Type::I64])
}

/// Single-level page table mapper with 32-bit physical addresses
///
/// This variant uses 4-byte page table entries for 32-bit physical addresses,
/// supporting up to 4 GiB of physical memory while maintaining 64-bit virtual addresses.
///
/// # Arguments
/// - `module`: WebAssembly module
/// - `f`: Function body being built
/// - `block`: Current block
/// - `vaddr`: Virtual address value (64-bit)
/// - `page_table_base`: Base address of page table (runtime Value or static u64)
/// - `memory`: Memory index to use for loads
///
/// # Returns
/// Physical address value (64-bit, but value fits in 32 bits)
pub fn standard_page_table_mapper_32(
    module: &mut Module,
    f: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    page_table_base: impl Into<PageTableBase>,
    memory: Memory,
) -> Value {
    let pt_base_value = page_table_base.into().to_value(f, block);
    // Extract page number: vaddr >> 16
    let shift_16 = f.add_op(block, Operator::I64Const { value: 16 }, &[], &[Type::I64]);
    let page_num = f.add_op(block, Operator::I64ShrU, &[vaddr, shift_16], &[Type::I64]);
    
    // Multiply page_num by 4 (size of u32 entry)
    let shift_2 = f.add_op(block, Operator::I64Const { value: 2 }, &[], &[Type::I64]);
    let entry_offset = f.add_op(block, Operator::I64Shl, &[page_num, shift_2], &[Type::I64]);
    
    // Add page table base address
    let entry_addr = f.add_op(block, Operator::I64Add, &[pt_base_value, entry_offset], &[Type::I64]);
    
    // Load 32-bit physical page base from page table and extend to 64-bit
    let phys_page_32 = f.add_op(
        block,
        Operator::I32Load {
            memory: MemoryArg {
                offset: 0,
                align: 2,
                memory,
            }
        },
        &[entry_addr],
        &[Type::I32]
    );
    let phys_page = f.add_op(block, Operator::I64ExtendI32U, &[phys_page_32], &[Type::I64]);
    
    // Extract page offset: vaddr & 0xFFFF
    let mask = f.add_op(block, Operator::I64Const { value: 0xFFFF }, &[], &[Type::I64]);
    let page_offset = f.add_op(block, Operator::I64And, &[vaddr, mask], &[Type::I64]);
    
    // Combine: phys_page + page_offset
    f.add_op(block, Operator::I64Add, &[phys_page, page_offset], &[Type::I64])
}

/// Multi-level page table mapper with 32-bit physical addresses
///
/// This variant uses 4-byte page table entries for 32-bit physical addresses,
/// supporting up to 4 GiB of physical memory in a 3-level page table structure.
///
/// # Arguments
/// - `module`: WebAssembly module
/// - `f`: Function body being built
/// - `block`: Current block
/// - `vaddr`: Virtual address value (64-bit)
/// - `l3_table_base`: Base address of level 3 page table (runtime Value or static u64)
/// - `memory`: Memory index to use for loads
///
/// # Returns
/// Physical address value (64-bit, but value fits in 32 bits)
pub fn multilevel_page_table_mapper_32(
    module: &mut Module,
    f: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    l3_table_base: impl Into<PageTableBase>,
    memory: Memory,
) -> Value {
    let l3_base_value = l3_table_base.into().to_value(f, block);
    // Helper to extract a 16-bit field from vaddr
    let extract_16bit = |f: &mut FunctionBody, block, val: Value, shift_amt: u64| -> Value {
        let shift = f.add_op(block, Operator::I64Const { value: shift_amt }, &[], &[Type::I64]);
        let shifted = f.add_op(block, Operator::I64ShrU, &[val, shift], &[Type::I64]);
        let mask = f.add_op(block, Operator::I64Const { value: 0xFFFF }, &[], &[Type::I64]);
        f.add_op(block, Operator::I64And, &[shifted, mask], &[Type::I64])
    };
    
    // Helper to load u32 and extend to u64
    let load_u32_extend = |f: &mut FunctionBody, block, addr: Value| -> Value {
        let val_32 = f.add_op(
            block,
            Operator::I32Load { memory: MemoryArg { offset: 0, align: 2, memory } },
            &[addr],
            &[Type::I32]
        );
        f.add_op(block, Operator::I64ExtendI32U, &[val_32], &[Type::I64])
    };
    
    let shift_2 = f.add_op(block, Operator::I64Const { value: 2 }, &[], &[Type::I64]);
    
    // Level 3: bits [63:48]
    let l3_idx = extract_16bit(f, block, vaddr, 48);
    let l3_offset = f.add_op(block, Operator::I64Shl, &[l3_idx, shift_2], &[Type::I64]);
    let l3_entry_addr = f.add_op(block, Operator::I64Add, &[l3_base_value, l3_offset], &[Type::I64]);
    let l2_table_base = load_u32_extend(f, block, l3_entry_addr);
    
    // Level 2: bits [47:32]
    let l2_idx = extract_16bit(f, block, vaddr, 32);
    let l2_offset = f.add_op(block, Operator::I64Shl, &[l2_idx, shift_2], &[Type::I64]);
    let l2_entry_addr = f.add_op(block, Operator::I64Add, &[l2_table_base, l2_offset], &[Type::I64]);
    let l1_table_base = load_u32_extend(f, block, l2_entry_addr);
    
    // Level 1: bits [31:16]
    let l1_idx = extract_16bit(f, block, vaddr, 16);
    let l1_offset = f.add_op(block, Operator::I64Shl, &[l1_idx, shift_2], &[Type::I64]);
    let l1_entry_addr = f.add_op(block, Operator::I64Add, &[l1_table_base, l1_offset], &[Type::I64]);
    let phys_page = load_u32_extend(f, block, l1_entry_addr);
    
    // Page offset: bits [15:0]
    let page_offset = extract_16bit(f, block, vaddr, 0);
    
    // Combine: phys_page + page_offset
    f.add_op(block, Operator::I64Add, &[phys_page, page_offset], &[Type::I64])
}

#[derive(Clone, Copy)]
pub struct Opts {
    pub mem: Memory,
    pub table: Table,
    pub ecall: Func,
    pub mapper: Option<(Func, Type)>,
    pub inline_ecall: bool,
}
impl Opts {
    fn map(
        &self,
        module: &Module,
        f: &mut FunctionBody,
        block: Block,
        v: Value,
        user: &mut [Value],
    ) -> Value {
        match self.mapper.as_ref().cloned() {
            None => v,
            Some((a, b)) => {
                let o = f.add_op(
                    block,
                    Operator::Call { function_index: a },
                    &user.iter().cloned().chain([v]).collect_vec(),
                    match &module.signatures[module.funcs[a].sig()] {
                        SignatureData::Func {
                            params,
                            returns,
                            shared,
                        } => &*returns,
                        _ => unreachable!(),
                    },
                );
                let o = results_ref_2(f, o);
                if o.len() != user.len() + 1 {
                    o[0]
                } else {
                    user.copy_from_slice(&o[1..]);
                    o[0]
                }
            }
        }
    }
}

/// Internal helper function for compiling RISC-V code with optional HINT handler.
///
/// This function contains the shared implementation used by both `compile` and
/// `compile_with_hints`. The `hint_handler` parameter is `Option` - when `None`,
/// HINT processing is skipped.
fn compile_internal(
    m: &mut Module<'_>,
    user: Vec<Type>,
    code: InputRef<'_>,
    start: u64,
    opts: Opts,
    tune: &Tunables,
    user_prepa: &mut (dyn FnMut(&mut Regs, &mut Value) + '_),
    retty: impl Iterator<Item = Type>,
    mut hint_handler: Option<&mut (dyn HintHandler + '_)>,
) -> Func {
    let n = tune.n;
    let _bleed = tune.bleed;
    // Base signature: user types + 31 GPRs (x1-x31) + 32 FPRs (f0-f31)
    // GPRs are I64, FPRs are F64 to support double-precision (D extension)
    // 
    // RISC-V Specification Quote:
    // "The D extension widens the 32 floating-point registers, f0–f31, to 64 bits."
    let base = user.iter().cloned().chain([Type::I64; 31]).chain([Type::F64; 32]);
    let mut code_fns: Vec<Func> = vec![];
    let j_sig = new_sig(
        m,
        portal_pc_waffle::SignatureData::Func {
            params: base.clone().chain([Type::I64]).collect(),
            returns: retty.collect(),
            shared: true,
        },
    );
    let j = m
        .funcs
        .push(portal_pc_waffle::FuncDecl::Import(j_sig, format!("r5_jmp")));
    let f_sig = j_sig;
    let pages = code
        .code
        .windows(4)
        .enumerate()
        .filter_map(|(i, j)| Some((i as u64 + start, u32::from_le_bytes(j.try_into().ok()?))));
    
    for (_i, page) in tune.paged_chunks(pages).enumerate() {
        let mut f = FunctionBody::new(&m, f_sig);
        let mut page = page.peekable();
        let Some((this_start, _)) = page.peek().cloned() else {
            continue;
        };
        let instrs: Vec<(u64, u32, BlockTarget)> = page
            .map(|(h, i)| {
                (h, i, {
                    let k = f.add_block();
                    BlockTarget {
                        block: k,
                        args: base
                            .clone()
                            .chain([Type::I64])
                            .map(|t| f.add_blockparam(k, t))
                            .collect(),
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut args = f.blocks[f.entry].params.iter().map(|a| a.1).collect_vec();
        let jt = args.pop().unwrap();
        let jt_this_page = f.add_op(
            f.entry,
            Operator::I64Const { value: this_start },
            &[],
            &[Type::I64],
        );
        let jt_root = f.add_op(
            f.entry,
            Operator::I64Const { value: start },
            &[],
            &[Type::I64],
        );
        let jt_this_page = f.add_op(f.entry, Operator::I64Sub, &[jt, jt_this_page], &[Type::I64]);
        let _jt_root = f.add_op(f.entry, Operator::I64Sub, &[jt, jt_root], &[Type::I64]);
        let s = f.add_block();
        let fail = f.add_block();
        f.set_terminator(fail, portal_pc_waffle::Terminator::UB);
        f.set_terminator(
            f.entry,
            portal_pc_waffle::Terminator::Select {
                value: jt_this_page,
                targets: instrs
                    .iter()
                    .enumerate()
                    .map(|(idx, a)| {
                        match *code.nj.get((a.0 - start) as usize).unwrap() && idx >= 4 + tune.bleed
                        {
                            false => BlockTarget {
                                block: a.2.block,
                                args: args.iter().cloned().chain(once(jt)).collect(),
                            },
                            true => BlockTarget {
                                block: fail,
                                args: vec![],
                            },
                        }
                    })
                    .collect(),
                default: BlockTarget {
                    block: s,
                    args: vec![],
                },
            },
        );
        f.set_terminator(
            s,
            portal_pc_waffle::Terminator::ReturnCall {
                func: j,
                args: args.iter().cloned().chain(once(jt)).collect(),
            },
        );
        for (ri, (h, i, BlockTarget { block: orig_block, mut args })) in instrs.iter().cloned().enumerate() {
            let mut block = orig_block;
            let rpc = args.pop().unwrap();
            let Some(inst) = Inst::decode_normal(i, rv_asm::Xlen::Rv64).ok() else {
                let r = f.add_op(block, Operator::I32Const { value: 1 }, &[], &[Type::I32]);
                f.set_terminator(
                    block,
                    portal_pc_waffle::Terminator::Return {
                        values: args.into_iter().chain([rpc]).chain(once(r)).collect(),
                    },
                );
                continue;
            };
            let rbs: [Block; 8193] = core::array::from_fn(|a| {
                if a < 4096 {
                    match instrs.get(ri.wrapping_sub(a * 2)) {
                        Some(a) => a.2.block,
                        None => f.entry,
                    }
                } else if a > 4096 {
                    match instrs.get(ri.wrapping_add((a - 4096) * 2)) {
                        Some(a) => a.2.block,
                        None => f.entry,
                    }
                } else {
                    orig_block
                }
            });
            let mut args_iter = args.drain(..);
            let mut uregs = Regs {
                user: user.iter().filter_map(|_x| args_iter.next()).collect(),
                gpr: args_iter.next_array().unwrap(),
                fpr: args_iter.next_array().unwrap(),
            };
            
            // Check for HINT instruction and invoke handler if provided
            // The handler can modify `block` to support complex branching
            if let Some(ref mut handler) = hint_handler {
                if let Some(marker) = detect_hint(&inst) {
                    let hint_info = HintInfo { marker, pc: h };
                    let mut ctx = HintCallbackContext {
                        hint: hint_info,
                        instruction: &inst,
                        module: m,
                        function: &mut f,
                        block: &mut block,
                        regs: &mut uregs,
                        pc_value: rpc,
                    };
                    handler.on_hint(&mut ctx);
                    // block may have been updated by the handler for complex branching
                }
            }
            
            // Use the potentially updated block for subsequent operations
            let orpc = rpc;
            let x = f.add_op(block, Operator::I64Const { value: 4 }, &[], &[Type::I64]);
            let rpc = f.add_op(block, Operator::I64Add, &[rpc, x], &[Type::I64]);
            compile_one(
                m,
                &mut f,
                &mut uregs,
                RiscVContext {
                    block,
                    rbs,
                    pc_value: rpc,
                    original_pc_value: orpc,
                    pc: h,
                    local_instruction_index: ri,
                    opts,
                },
                inst,
                &instrs,
                code.nest(),
                j,
            );
        }
        let f = m.funcs.push(portal_pc_waffle::FuncDecl::Body(
            f_sig,
            format!("r5_slice"),
            f,
        ));
        code_fns.push(f);
    }
    
    let ti = m.tables[opts.table].func_elements.as_mut().unwrap().len() as u64;
    m.tables[opts.table]
        .func_elements
        .as_mut()
        .unwrap()
        .extend(code_fns.drain(..));
    let mut f = FunctionBody::new(&m, j_sig);
    let mut args = f.blocks[f.entry].params.iter().map(|a| a.1).collect_vec();
    let jt = args.pop().unwrap();
    let jt_root = f.add_op(
        f.entry,
        Operator::I64Const { value: start },
        &[],
        &[Type::I64],
    );
    let jt_root = f.add_op(f.entry, Operator::I64Sub, &[jt, jt_root], &[Type::I64]);
    let jc = f.add_op(
        f.entry,
        Operator::I64Const {
            value: code.len() as u64,
        },
        &[],
        &[Type::I64],
    );
    let jc = f.add_op(f.entry, Operator::I64LtU, &[jt_root, jc], &[Type::I32]);
    let s = f.add_block();
    let t = {
        let mut args = args.clone();
        let mut args = args.drain(..);
        let mut uregs = Regs {
            user: user.iter().filter_map(|_x| args.next()).collect(),
            gpr: args.next_array().unwrap(),
            fpr: args.next_array().unwrap(),
        };
        let mut rpc = jt;
        user_prepa(&mut uregs, &mut rpc);
        BlockTarget {
            block: f.entry,
            args: uregs.to_args().into_iter().chain([rpc]).collect(),
        }
    };
    f.set_terminator(
        f.entry,
        portal_pc_waffle::Terminator::CondBr {
            cond: jc,
            if_true: BlockTarget {
                block: s,
                args: vec![],
            },
            if_false: t,
        },
    );
    let sv = f.add_op(s, Operator::I64Const { value: n as u64 }, &[], &[Type::I64]);
    let sv = f.add_op(s, Operator::I64DivU, &[jt_root, sv], &[Type::I64]);
    let sv2 = f.add_op(s, Operator::I64Const { value: ti }, &[], &[Type::I64]);
    let sv = f.add_op(s, Operator::I64Add, &[sv2, sv], &[Type::I64]);
    f.set_terminator(
        s,
        portal_pc_waffle::Terminator::ReturnCallIndirect {
            sig: f_sig,
            table: opts.table,
            args: args.into_iter().chain([jt, sv]).collect(),
        },
    );
    m.funcs[j] = FuncDecl::Body(j_sig, format!("r5_jmp"), f);
    return j;
}

/// Compile RISC-V code to WebAssembly.
///
/// This function compiles RISC-V binary code into WebAssembly, creating the
/// necessary function bodies and table entries for execution.
///
/// For HINT instruction processing during compilation, use [`compile_with_hints`]
/// instead.
pub fn compile(
    m: &mut Module,
    user: Vec<Type>,
    code: InputRef<'_>,
    start: u64,
    opts: Opts,
    tune: &Tunables,
    user_prepa: &mut (dyn FnMut(&mut Regs, &mut Value) + '_),
    retty: impl Iterator<Item = Type>,
) -> Func {
    compile_internal(m, user, code, start, opts, tune, user_prepa, retty, None)
}

/// Compile RISC-V code to WebAssembly with inline HINT processing.
///
/// This is an extended version of [`compile`] that supports inline processing
/// of HINT instructions through a handler. When a HINT instruction
/// (`addi x0, x0, N` where N != 0) is encountered, the provided handler is invoked.
///
/// This is useful for:
/// - Test case boundary detection during compilation
/// - Instrumenting generated code at HINT locations
/// - Collecting HINT markers during compilation
///
/// # Arguments
///
/// * `m` - The WebAssembly module to add functions to
/// * `user` - User-defined types for the function signature
/// * `code` - The RISC-V binary code to compile
/// * `start` - The starting address of the code
/// * `opts` - Compilation options
/// * `tune` - Tuning parameters for compilation
/// * `user_prepa` - User preparation callback for register/PC initialization
/// * `retty` - Return type iterator for the compiled function
/// * `hint_handler` - Handler invoked for each HINT instruction
///
/// # Example
///
/// ```ignore
/// let mut hints_found = Vec::new();
/// let func = compile_with_hints(
///     &mut module,
///     vec![],
///     code.as_ref(),
///     start_addr,
///     opts,
///     &tune,
///     &mut |_, _| {},
///     std::iter::repeat(Type::I64).take(33),
///     &mut |ctx: &mut HintCallbackContext| {
///         hints_found.push(ctx.hint.clone());
///     },
/// );
/// ```
pub fn compile_with_hints(
    m: &mut Module<'_>,
    user: Vec<Type>,
    code: InputRef<'_>,
    start: u64,
    opts: Opts,
    tune: &Tunables,
    user_prepa: &mut (dyn FnMut(&mut Regs, &mut Value) + '_),
    retty: impl Iterator<Item = Type>,
    hint_handler: &mut (dyn HintHandler + '_),
) -> Func {
    compile_internal(m, user, code, start, opts, tune, user_prepa, retty, Some(hint_handler))
}

/// Scan code for HINT instructions and return a list of detected HINTs.
///
/// This function scans through RISC-V binary code looking for HINT markers
/// (specifically `addi x0, x0, N` where N != 0) which are used by rv-corpus
/// to mark test case boundaries.
///
/// # Arguments
///
/// * `code` - The binary code bytes to scan
/// * `start` - The starting address of the code
/// * `xlen` - The RISC-V xlen (Rv32 or Rv64)
///
/// # Returns
///
/// A vector of `HintInfo` structs, each containing the marker value and PC
/// address where the HINT was found.
pub fn scan_hints(code: &[u8], start: u64, xlen: rv_asm::Xlen) -> Vec<HintInfo> {
    let mut hints = Vec::new();
    
    for (offset, window) in code.windows(4).enumerate() {
        if let Ok(bytes) = window.try_into() {
            let instruction = u32::from_le_bytes(bytes);
            if let Ok(inst) = Inst::decode_normal(instruction, xlen) {
                if let Some(marker) = detect_hint(&inst) {
                    hints.push(HintInfo {
                        marker,
                        pc: start + offset as u64,
                    });
                }
            }
        }
    }
    
    hints
}

/// Scan code for all HINT instructions (including NOP) and return details.
///
/// This is a more comprehensive version of `scan_hints` that includes
/// all instructions that qualify as HINTs according to the RISC-V specification,
/// including the canonical NOP (`addi x0, x0, 0`).
///
/// # Arguments
///
/// * `code` - The binary code bytes to scan
/// * `start` - The starting address of the code
/// * `xlen` - The RISC-V xlen (Rv32 or Rv64)
///
/// # Returns
///
/// A vector of tuples containing (pc, instruction, marker) where:
/// - `pc` is the address of the instruction
/// - `instruction` is the decoded `Inst`
/// - `marker` is `Some(marker_value)` if this is a test case marker, `None` otherwise
pub fn scan_all_hints(
    code: &[u8],
    start: u64,
    xlen: rv_asm::Xlen,
) -> Vec<(u64, Inst, Option<i32>)> {
    let mut hints = Vec::new();
    
    for (offset, window) in code.windows(4).enumerate() {
        if let Ok(bytes) = window.try_into() {
            let instruction = u32::from_le_bytes(bytes);
            if let Ok(inst) = Inst::decode_normal(instruction, xlen) {
                if is_hint_instruction(&inst) {
                    let marker = detect_hint(&inst);
                    hints.push((start + offset as u64, inst, marker));
                }
            }
        }
    }
    
    hints
}
