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
use rv_asm::{Inst, Reg};
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
pub struct HintCallbackContext<'a, 'b> {
    /// The HINT information (marker value and PC)
    pub hint: HintInfo,
    /// The decoded HINT instruction
    pub instruction: &'a Inst,
    /// The WebAssembly module being constructed
    pub module: &'a mut Module<'b>,
    /// The current function body being built
    pub function: &'a mut FunctionBody,
    /// The current block where the HINT was encountered
    pub block: Block,
    /// The current register state
    pub regs: &'a mut Regs,
    /// The PC value as a WebAssembly Value
    pub pc_value: Value,
}

/// Type alias for the HINT callback function.
///
/// The callback is invoked for each HINT instruction encountered during
/// compilation when `Opts::process_hints` is enabled and a callback is provided.
///
/// The callback receives a `HintCallbackContext` which provides access to:
/// - The HINT marker value and PC
/// - The decoded instruction
/// - The WebAssembly module and function being built
/// - The current block and register state
///
/// The lifetime parameters ensure that:
/// - `'a` is the lifetime of the callback reference itself
/// - `'b` is the lifetime bound on the callback trait object
/// - The context's inner lifetimes (`'_`, `'_`) are anonymous to prevent data leakage
///
/// # Example
///
/// ```ignore
/// let mut hints_found = Vec::new();
/// let hint_callback = |ctx: &mut HintCallbackContext| {
///     hints_found.push(ctx.hint.clone());
///     // Optionally add WebAssembly instructions here
/// };
/// ```
pub type HintCallback<'a, 'b> = &'a mut (dyn FnMut(&mut HintCallbackContext<'_, '_>) + 'b);

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

pub struct Regs {
    pub gpr: [Value; 31],
    pub user: Vec<Value>,
}
impl Regs {
    pub fn from_args<T>(
        user: &[T],
        mut args: &mut (dyn Iterator<Item = Value> + '_),
    ) -> Option<Self> {
        Some(Regs {
            user: user.iter().filter_map(|x| args.next()).collect(),
            gpr: {
                let mut args: &mut &mut _ = &mut args;
                args.next_array()?
            },
        })
    }
    pub fn to_args(&self) -> impl Iterator<Item = Value> {
        return self.user.iter().chain(self.gpr.iter()).cloned();
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
                        Inst::Fence { fence } => {
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
#[derive(Clone, Copy)]
pub struct Opts {
    pub mem: Memory,
    pub table: Table,
    pub ecall: Func,
    pub mapper: Option<(Func, Type)>,
    pub inline_ecall: bool,
    /// Enable processing of HINT instructions from rv-corpus test suite.
    ///
    /// When enabled, HINT instructions (specifically `addi x0, x0, N` where N != 0)
    /// will be detected during compilation. This is useful for testing and debugging
    /// with the rv-corpus test binaries which use HINT markers to identify test cases.
    ///
    /// When disabled (default), HINT instructions are treated as regular no-ops.
    pub process_hints: bool,
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

/// Internal helper function for compiling RISC-V code with optional HINT callback.
///
/// This function contains the shared implementation used by both `compile` and
/// `compile_with_hints`. The `hint_callback` parameter is `Option` - when `None`,
/// HINT processing is skipped even if `opts.process_hints` is true.
fn compile_internal(
    m: &mut Module<'_>,
    user: Vec<Type>,
    code: InputRef<'_>,
    start: u64,
    opts: Opts,
    tune: &Tunables,
    user_prepa: &mut (dyn FnMut(&mut Regs, &mut Value) + '_),
    retty: impl Iterator<Item = Type>,
    mut hint_callback: Option<&mut (dyn FnMut(&mut HintCallbackContext<'_, '_>) + '_)>,
) -> Func {
    let n = tune.n;
    let _bleed = tune.bleed;
    let base = user.iter().cloned().chain([Type::I64; 31]);
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
        for (ri, (h, i, BlockTarget { block, mut args })) in instrs.iter().cloned().enumerate() {
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
                    block
                }
            });
            let mut args_iter = args.drain(..);
            let mut uregs = Regs {
                user: user.iter().filter_map(|_x| args_iter.next()).collect(),
                gpr: args_iter.next_array().unwrap(),
            };
            
            // Check for HINT instruction and invoke callback if enabled
            if let Some(ref mut callback) = hint_callback {
                if opts.process_hints {
                    if let Some(marker) = detect_hint(&inst) {
                        let hint_info = HintInfo { marker, pc: h };
                        let mut ctx = HintCallbackContext {
                            hint: hint_info,
                            instruction: &inst,
                            module: m,
                            function: &mut f,
                            block,
                            regs: &mut uregs,
                            pc_value: rpc,
                        };
                        callback(&mut ctx);
                    }
                }
            }
            
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
/// of HINT instructions through a callback. When `Opts::process_hints` is enabled
/// and a HINT instruction (`addi x0, x0, N` where N != 0) is encountered, the
/// provided callback is invoked.
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
/// * `opts` - Compilation options (including `process_hints` flag)
/// * `tune` - Tuning parameters for compilation
/// * `user_prepa` - User preparation callback for register/PC initialization
/// * `retty` - Return type iterator for the compiled function
/// * `hint_callback` - Callback invoked for each HINT instruction when `process_hints` is true
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
///     Opts { process_hints: true, ..opts },
///     &tune,
///     &mut |_, _| {},
///     std::iter::repeat(Type::I64).take(33),
///     &mut |ctx| {
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
    hint_callback: &mut (dyn FnMut(&mut HintCallbackContext<'_, '_>) + '_),
) -> Func {
    compile_internal(m, user, code, start, opts, tune, user_prepa, retty, Some(hint_callback))
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
