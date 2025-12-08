# Paging System Implementation - rift

See `r5-abi-specs/PAGING.md` for the complete paging specification.

## rift-Specific Implementation

**Target:** RISC-V to WebAssembly compilation

**API Functions:**
- `standard_page_table_mapper()` - Single-level 64KB paging (64-bit physical)
- `standard_page_table_mapper_32()` - Single-level 64KB paging (32-bit physical)
- `multilevel_page_table_mapper()` - 3-level hierarchical paging (64-bit physical)
- `multilevel_page_table_mapper_32()` - 3-level hierarchical paging (32-bit physical)

**Integration:**
Uses `Opts.mapper` callback to inject custom address translation into compiled WebAssembly functions.

**Example:**
```rust
// With static constant address
let phys_addr = standard_page_table_mapper(
    module, function, block, vaddr,
    0x1000000u64,  // static constant
    &regs,
    memory
);

// With value from Regs.page_table_base field
let phys_addr = standard_page_table_mapper(
    module, function, block, vaddr,
    PageTableBase::FromRegs,
    &regs,
    memory
);

// With WebAssembly global variable
let phys_addr = standard_page_table_mapper(
    module, function, block, vaddr,
    global_index,  // Global
    &regs,
    memory
);

// 32-bit physical addresses (4 GiB limit)
let phys_addr = standard_page_table_mapper_32(
    module, function, block, vaddr,
    PageTableBase::FromRegs,
    &regs,
    memory
);
```

See `src/lib.rs` for implementation details.
