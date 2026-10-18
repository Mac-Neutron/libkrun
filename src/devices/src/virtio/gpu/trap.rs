//! Trap blobs (MacNeutron, Direct3D forwarding): a dxmt-context blob whose id has bit 63 set is
//! left unmapped, so a guest store to it exits to the storing vCPU's thread, which hands the stored
//! value to the VM host's handler before the MMIO bus. The guest flushes a command ring this way
//! and the host runs it on that vCPU, with no thread spinning. The ranges are read lock-free on
//! every MMIO exit: the lock only orders the gpu worker's changes.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

use log::warn;

pub const TRAP_BLOB_ID: u64 = 1 << 63;

type Handler = extern "C" fn(ctx_id: u32, value: u64, opaque: *mut c_void);

struct Range {
    start: AtomicU64,
    end: AtomicU64,
    ctx_id: AtomicU32,
}

// One trap blob per guest process that forwards Direct3D.
static RANGES: [Range; 64] = [const {
    Range {
        start: AtomicU64::new(0),
        end: AtomicU64::new(0),
        ctx_id: AtomicU32::new(0),
    }
}; 64];
static CHANGES: Mutex<()> = Mutex::new(());
static HANDLER: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static OPAQUE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

pub fn add(start: u64, size: u64, ctx_id: u32) -> bool {
    let _changes = CHANGES.lock().unwrap();
    let Some(range) = RANGES
        .iter()
        .find(|r| r.start.load(Ordering::Relaxed) == 0)
    else {
        warn!("trap: more than {} trap blobs", RANGES.len());
        return false;
    };
    range.ctx_id.store(ctx_id, Ordering::Relaxed);
    range.end.store(start + size, Ordering::Relaxed);
    range.start.store(start, Ordering::Release);
    true
}

pub fn remove(start: u64) {
    let _changes = CHANGES.lock().unwrap();
    for range in RANGES.iter().filter(|r| r.start.load(Ordering::Relaxed) == start) {
        range.start.store(0, Ordering::Release);
    }
}

fn ctx_of(addr: u64) -> Option<u32> {
    RANGES.iter().find_map(|r| {
        let start = r.start.load(Ordering::Acquire);
        let inside = start != 0 && addr >= start && addr < r.end.load(Ordering::Relaxed);
        inside.then(|| r.ctx_id.load(Ordering::Relaxed))
    })
}

pub fn contains(addr: u64) -> bool {
    ctx_of(addr).is_some()
}

/// A guest store at `addr`: true if it was a trap blob's, which the handler has run.
pub fn write(addr: u64, data: &[u8]) -> bool {
    let Some(ctx_id) = ctx_of(addr) else {
        return false;
    };
    let handler = HANDLER.load(Ordering::Acquire);
    match <[u8; 8]>::try_from(data) {
        Ok(value) if !handler.is_null() => {
            // SAFETY: krun_set_trap_handler stored a Handler.
            let f: Handler = unsafe { std::mem::transmute::<*mut c_void, Handler>(handler) };
            f(ctx_id, u64::from_le_bytes(value), OPAQUE.load(Ordering::Relaxed));
        }
        _ => warn!(
            "trap: a {}-byte store on context {ctx_id}: not 8 bytes, or no handler",
            data.len()
        ),
    }
    true
}

/// Sets the function a trap blob's 8-byte stores call (on the storing vCPU's thread), before the VM
/// starts.
///
/// # Safety
/// `f` is NULL or callable from any vCPU thread with `opaque`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn krun_set_trap_handler(f: Option<Handler>, opaque: *mut c_void) -> i32 {
    OPAQUE.store(opaque, Ordering::Relaxed);
    HANDLER.store(f.map_or(std::ptr::null_mut(), |f| f as *mut c_void), Ordering::Release);
    0
}
