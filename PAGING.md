# Paging System Implementation - rift

See `r5-abi-specs/PAGING.md` for the complete paging specification.

## rift-Specific Implementation

**Target:** RISC-V to WebAssembly compilation

**API Functions:**
- `standard_page_table_mapper()` - Single-level 64KB paging
- `multilevel_page_table_mapper()` - 3-level hierarchical paging

**Integration:**
Uses `Opts.mapper` callback to inject custom address translation into compiled WebAssembly functions.

**Example:**
```rust
let opts = Opts {
    mapper: Some((mapper_func, mapper_type)),
    // ... other options
};
```

See `src/lib.rs` for implementation details.
