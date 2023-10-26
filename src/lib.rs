pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod apply_normalisation;


#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
