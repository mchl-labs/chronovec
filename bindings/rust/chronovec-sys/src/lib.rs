//! Raw FFI declarations for the ChronoVec C ABI.
//!
//! Mirrors the bundled C ABI header exactly. Prefer the safe `chronovec`
//! crate; this exists so the safe layer has nothing hidden underneath it.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct cv_index {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct cv_stats {
    pub clock: u64,
    pub pages: u64,
    pub live_vectors: u64,
    pub retained_versions: u64,
    pub allocated_slots: u64,
    pub splits: u64,
    pub reclaimed_versions: u64,
    pub merges: u64,
    pub routing_full_rebuilds: u64,
    pub routing_incremental_updates: u64,
    pub routing_centroid_bytes: u64,
    pub routing_edge_count: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct cv_search_metrics {
    pub directory_pages: u64,
    pub centroid_scores: u64,
    pub routed_pages: u64,
    pub physical_candidates: u64,
    pub visible_candidates: u64,
    pub mvcc_filtered: u64,
    pub sketch_scores: u64,
    pub exact_scores: u64,
    pub result_count: u64,
    pub bound_pruned_pages: u64,
    pub preparation_ns: u64,
    pub routing_ns: u64,
    pub screening_ns: u64,
    pub rerank_ns: u64,
    pub result_materialization_ns: u64,
    pub total_ns: u64,
}

pub const CV_COSINE: c_int = 0;
pub const CV_L2: c_int = 1;
pub const CV_ENABLE_INT8_SCREENING: u32 = 1;
pub const CV_ENABLE_ADAPTIVE_BOUNDS: u32 = 2;
pub const CV_SEARCH_ADAPTIVE: u32 = 1;
pub const CV_SEARCH_LINEAR_ROUTING: u32 = 2;

extern "C" {
    pub fn cv_create(
        dimensions: usize,
        metric: c_int,
        page_capacity: usize,
        nprobe: usize,
    ) -> *mut cv_index;
    pub fn cv_create_with_options(
        dimensions: usize,
        metric: c_int,
        page_capacity: usize,
        nprobe: usize,
        flags: u32,
        rerank_factor: usize,
    ) -> *mut cv_index;
    pub fn cv_destroy(index: *mut cv_index);
    pub fn cv_insert(
        index: *mut cv_index,
        id: i64,
        vector: *const f32,
        timestamp: u64,
        committed: *mut u64,
    ) -> c_int;
    pub fn cv_delete(index: *mut cv_index, id: i64, timestamp: u64, committed: *mut u64) -> c_int;
    pub fn cv_search(
        index: *mut cv_index,
        query: *const f32,
        k: usize,
        snapshot: u64,
        nprobe: usize,
        out_ids: *mut i64,
        out_distances: *mut f32,
    ) -> usize;
    pub fn cv_search_with_options(
        index: *mut cv_index,
        query: *const f32,
        k: usize,
        snapshot: u64,
        nprobe: usize,
        search_flags: u32,
        out_ids: *mut i64,
        out_distances: *mut f32,
    ) -> usize;
    pub fn cv_vacuum(index: *mut cv_index, oldest_snapshot: u64, budget_versions: usize) -> usize;
    pub fn cv_save_checkpoint(index: *mut cv_index, path: *const c_char) -> c_int;
    pub fn cv_load_checkpoint(path: *const c_char) -> *mut cv_index;
    pub fn cv_dimensions(index: *const cv_index) -> usize;
    pub fn cv_metric(index: *const cv_index) -> c_int;
    pub fn cv_page_capacity(index: *const cv_index) -> usize;
    pub fn cv_default_nprobe(index: *const cv_index) -> usize;
    pub fn cv_clock(index: *const cv_index) -> u64;
    pub fn cv_get_stats(index: *const cv_index, out: *mut cv_stats) -> c_int;
    pub fn cv_get_last_search_metrics(out: *mut cv_search_metrics) -> c_int;
    pub fn cv_last_error() -> *const c_char;
}
