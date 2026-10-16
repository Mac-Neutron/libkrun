//! MacNeutron's stand-in for rutabaga_gfx 0.1.85: the subset of its API that libkrun's virtio-gpu device uses, with
//! one context type, dxmt (capset 24). Blobs are host memory the VM host's executor shares with the guest; command
//! buffers and fences go to the callbacks it registers with `mn_dxmt_set_ops`, and it completes fences with
//! `mn_dxmt_fence_done`. A freed blob's memory goes to the executor too, which unmaps it once nothing uses it. No virglrenderer, no scanouts: 2D, 3D and transfer calls succeed and do nothing.
//! macOS only (the export handle carries no descriptor).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::io::{IoSlice, IoSliceMut};
use std::path::PathBuf;
use std::ptr;
use std::sync::{Arc, Mutex};

use log::warn;

// Under 32: the guest driver builds its capset mask with `1 << id` on an int.
pub const MN_CAPSET_DXMT: u32 = 24;
// The host page size (Apple silicon): hv_vm_map needs blobs that start and end on it.
const BLOB_ALIGN: u64 = 16384;
const CONTEXT_INIT_CAPSET_ID_MASK: u32 = 0xff;

pub const VIRTIO_GPU_BLOB_MEM_HOST3D: u32 = 2;
pub const VIRTIO_GPU_BLOB_FLAG_USE_MAPPABLE: u32 = 1;

pub const RUTABAGA_CAPSET_VIRGL: u32 = 1;
pub const RUTABAGA_CAPSET_VIRGL2: u32 = 2;
pub const RUTABAGA_CAPSET_VENUS: u32 = 4;
pub const RUTABAGA_CAPSET_CROSS_DOMAIN: u32 = 5;
pub const RUTABAGA_CAPSET_DRM: u32 = 6;

pub const RUTABAGA_PIPE_TEXTURE_2D: u32 = 2;
pub const RUTABAGA_PIPE_BIND_RENDER_TARGET: u32 = 2;

pub const RUTABAGA_MAP_CACHE_MASK: u32 = 0x0f;
pub const RUTABAGA_MAP_CACHE_CACHED: u32 = 0x01;
pub const RUTABAGA_MAP_ACCESS_MASK: u32 = 0xf0;
pub const RUTABAGA_MAP_ACCESS_READ: u32 = 0x10;
pub const RUTABAGA_MAP_ACCESS_WRITE: u32 = 0x20;
pub const RUTABAGA_MAP_ACCESS_RW: u32 = 0x30;

pub const RUTABAGA_HANDLE_TYPE_MEM_OPAQUE_FD: u32 = 0x0001;
pub const RUTABAGA_HANDLE_TYPE_MEM_DMABUF: u32 = 0x0002;
pub const RUTABAGA_HANDLE_TYPE_MEM_SHM: u32 = 0x0004;

pub const RUTABAGA_PATH_TYPE_WAYLAND: u32 = 1;
pub const RUTABAGA_PATH_TYPE_X11: u32 = 2;
pub const RUTABAGA_PATH_TYPE_PIPEWIRE: u32 = 3;

#[derive(Debug)]
pub enum RutabagaError {
    InvalidCapset,
    InvalidContextId,
    InvalidResourceId,
    InvalidBlob(&'static str),
    Unsupported,
}

impl fmt::Display for RutabagaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RutabagaError::InvalidCapset => write!(f, "dxmt: not the dxmt capset"),
            RutabagaError::InvalidContextId => write!(f, "dxmt: invalid context id"),
            RutabagaError::InvalidResourceId => write!(f, "dxmt: invalid resource id"),
            RutabagaError::InvalidBlob(why) => write!(f, "dxmt: invalid blob: {why}"),
            RutabagaError::Unsupported => write!(f, "dxmt: unsupported"),
        }
    }
}

impl std::error::Error for RutabagaError {}

pub type RutabagaResult<T> = Result<T, RutabagaError>;

pub struct RutabagaDescriptor(#[allow(dead_code)] File);

impl From<File> for RutabagaDescriptor {
    fn from(file: File) -> Self {
        Self(file)
    }
}

pub trait VirtioFsLookup: Send + Sync {
    fn get_exported_descriptor(&self, fs_id: u64, handle: u64) -> RutabagaResult<RutabagaDescriptor>;
}

#[derive(Copy, Clone, Debug)]
pub struct RutabagaIovec {
    pub base: *mut c_void,
    pub len: usize,
}

pub struct RutabagaPath {
    pub path: PathBuf,
    pub path_type: u32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RutabagaFence {
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub ring_idx: u8,
}

#[derive(Clone)]
pub struct RutabagaFenceHandler(Arc<dyn Fn(RutabagaFence) + Send + Sync>);

impl RutabagaFenceHandler {
    pub fn new<F: Fn(RutabagaFence) + Send + Sync + 'static>(f: F) -> Self {
        Self(Arc::new(f))
    }

    pub fn call(&self, fence: RutabagaFence) {
        (self.0)(fence)
    }
}

pub enum RutabagaComponentType {
    Rutabaga2D,
    VirglRenderer,
    CrossDomain,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ResourceCreate3D {
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ResourceCreateBlob {
    pub blob_mem: u32,
    pub blob_flags: u32,
    pub blob_id: u64,
    pub size: u64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer3D {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
    pub level: u32,
    pub stride: u32,
    pub layer_stride: u32,
    pub offset: u64,
}

impl Transfer3D {
    pub fn new_2d(x: u32, y: u32, w: u32, h: u32, offset: u64) -> Self {
        Self { x, y, w, h, d: 1, offset, ..Default::default() }
    }
}

pub struct Resource3DInfo {
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub strides: [u32; 4],
    pub offsets: [u32; 4],
    pub modifier: u64,
}

pub struct RutabagaMapping {
    pub ptr: u64,
    pub size: u64,
}

pub struct MesaHandle {
    pub os_handle: Option<File>,
    pub handle_type: u32,
}

pub struct RutabagaHandle {
    handle_type: u32,
}

impl RutabagaHandle {
    pub fn as_mesa_handle(&self) -> Option<MesaHandle> {
        Some(MesaHandle { os_handle: None, handle_type: self.handle_type })
    }
}

/// The VM host's executor (C ABI). Any callback may be NULL.
#[repr(C)]
pub struct MnDxmtOps {
    pub ctx: *mut c_void,
    pub context_create: Option<extern "C" fn(*mut c_void, u32)>,
    pub context_destroy: Option<extern "C" fn(*mut c_void, u32)>,
    pub blob_created: Option<extern "C" fn(*mut c_void, u32, u64, *mut c_void, u64)>,
    /// The guest freed the blob: the mapping (host_ptr, size) is the executor's now, to unmap once nothing uses it.
    pub blob_destroyed: Option<extern "C" fn(*mut c_void, u32, u64, *mut c_void, u64)>,
    pub submit: Option<extern "C" fn(*mut c_void, u32, *const c_void, u32)>,
    pub fence: Option<extern "C" fn(*mut c_void, u32, u32, u64)>,
}

#[derive(Clone, Copy)]
struct Ops(MnDxmtOps);
// The executor's context is shared with its own threads; it says so by registering it.
unsafe impl Send for Ops {}

impl Clone for MnDxmtOps {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for MnDxmtOps {}

static OPS: Mutex<Option<Ops>> = Mutex::new(None);

struct FenceState {
    handler: RutabagaFenceHandler,
    pending: HashMap<(u32, u32, u64), RutabagaFence>,
}

// One gpu device per VM host process, so its fences are process-wide: the executor completes them from its threads.
static FENCES: Mutex<Option<FenceState>> = Mutex::new(None);

fn ops() -> Option<MnDxmtOps> {
    OPS.lock().unwrap().map(|o| o.0)
}

/// Registers the executor's callbacks (copied); NULL removes them. Without an executor, fences complete at once.
///
/// # Safety
/// `ops` is NULL or points to a valid `MnDxmtOps` whose callbacks stay callable with its `ctx` from any thread.
#[no_mangle]
pub unsafe extern "C" fn mn_dxmt_set_ops(ops: *const MnDxmtOps) {
    *OPS.lock().unwrap() = if ops.is_null() { None } else { Some(Ops(unsafe { *ops })) };
}

/// Completes a fence `ops.fence` was given.
#[no_mangle]
pub extern "C" fn mn_dxmt_fence_done(ctx_id: u32, ring_idx: u32, fence_id: u64) {
    let done = {
        let mut state = FENCES.lock().unwrap();
        state.as_mut().and_then(|s| {
            s.pending.remove(&(ctx_id, ring_idx, fence_id)).map(|f| (s.handler.clone(), f))
        })
    };
    match done {
        Some((handler, fence)) => handler.call(fence),
        None => warn!("dxmt: fence {fence_id} on context {ctx_id} ring {ring_idx} is not pending"),
    }
}

pub struct RutabagaBuilder {
    fence_handler: RutabagaFenceHandler,
}

impl RutabagaBuilder {
    pub fn new(_capset_mask: u64, fence_handler: RutabagaFenceHandler) -> Self {
        Self { fence_handler }
    }

    pub fn set_default_component(self, _component: RutabagaComponentType) -> Self {
        self
    }

    pub fn set_use_egl(self, _v: bool) -> Self {
        self
    }

    pub fn set_use_gles(self, _v: bool) -> Self {
        self
    }

    pub fn set_use_render_server(self, _v: bool) -> Self {
        self
    }

    pub fn set_rutabaga_paths(self, _paths: Option<Vec<RutabagaPath>>) -> Self {
        self
    }

    pub fn set_virtiofs_lookup(self, _lookup: Arc<dyn VirtioFsLookup>) -> Self {
        self
    }

    pub fn build(self) -> RutabagaResult<Rutabaga> {
        *FENCES.lock().unwrap() = Some(FenceState {
            handler: self.fence_handler.clone(),
            pending: HashMap::new(),
        });
        Ok(Rutabaga {
            fence_handler: self.fence_handler,
            contexts: HashSet::new(),
            plain: HashSet::new(),
            blobs: HashMap::new(),
        })
    }
}

struct Blob {
    ctx_id: u32,
    blob_id: u64,
    ptr: *mut c_void,
    size: u64,
}

pub struct Rutabaga {
    fence_handler: RutabagaFenceHandler,
    contexts: HashSet<u32>,
    plain: HashSet<u32>,
    blobs: HashMap<u32, Blob>,
}

// The blobs' memory is the executor's to share; the worker thread owns this table.
unsafe impl Send for Rutabaga {}

impl Rutabaga {
    pub fn get_capset_info(&self, index: u32) -> RutabagaResult<(u32, u32, u32)> {
        match index {
            0 => Ok((MN_CAPSET_DXMT, 0, 0)),
            _ => Err(RutabagaError::InvalidCapset),
        }
    }

    pub fn get_capset(&self, capset_id: u32, _version: u32) -> RutabagaResult<Vec<u8>> {
        match capset_id {
            MN_CAPSET_DXMT => Ok(Vec::new()),
            _ => Err(RutabagaError::InvalidCapset),
        }
    }

    pub fn force_ctx_0(&self) {}

    pub fn create_context(&mut self, ctx_id: u32, context_init: u32, _name: Option<&str>) -> RutabagaResult<()> {
        if context_init & CONTEXT_INIT_CAPSET_ID_MASK != MN_CAPSET_DXMT {
            return Err(RutabagaError::InvalidCapset);
        }
        if !self.contexts.insert(ctx_id) {
            return Err(RutabagaError::InvalidContextId);
        }
        if let Some(o) = ops() {
            if let Some(f) = o.context_create {
                f(o.ctx, ctx_id);
            }
        }
        Ok(())
    }

    pub fn destroy_context(&mut self, ctx_id: u32) -> RutabagaResult<()> {
        if !self.contexts.remove(&ctx_id) {
            return Err(RutabagaError::InvalidContextId);
        }
        if let Some(o) = ops() {
            if let Some(f) = o.context_destroy {
                f(o.ctx, ctx_id);
            }
        }
        Ok(())
    }

    pub fn context_attach_resource(&mut self, _ctx_id: u32, _resource_id: u32) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn context_detach_resource(&mut self, _ctx_id: u32, _resource_id: u32) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn submit_command(&mut self, ctx_id: u32, commands: &mut [u8], _fence_ids: &[u64]) -> RutabagaResult<()> {
        if !self.contexts.contains(&ctx_id) {
            return Err(RutabagaError::InvalidContextId);
        }
        if let Some(o) = ops() {
            if let Some(f) = o.submit {
                f(o.ctx, ctx_id, commands.as_ptr() as *const c_void, commands.len() as u32);
            }
        }
        Ok(())
    }

    pub fn create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        let executor = ops().and_then(|o| o.fence.map(|f| (o.ctx, f)));
        let Some((ctx, f)) = executor else {
            self.fence_handler.call(fence);
            return Ok(());
        };
        let key = (fence.ctx_id, fence.ring_idx as u32, fence.fence_id);
        if let Some(state) = FENCES.lock().unwrap().as_mut() {
            state.pending.insert(key, fence);
        }
        // Outside the lock: the executor may complete it from this call.
        f(ctx, key.0, key.1, key.2);
        Ok(())
    }

    pub fn resource_create_3d(&mut self, resource_id: u32, _info: ResourceCreate3D) -> RutabagaResult<()> {
        self.plain.insert(resource_id);
        Ok(())
    }

    pub fn resource_create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        info: ResourceCreateBlob,
        _iovecs: Option<Vec<RutabagaIovec>>,
        _handle: Option<MesaHandle>,
    ) -> RutabagaResult<()> {
        if !self.contexts.contains(&ctx_id) {
            return Err(RutabagaError::InvalidContextId);
        }
        if info.blob_mem != VIRTIO_GPU_BLOB_MEM_HOST3D || info.blob_flags & VIRTIO_GPU_BLOB_FLAG_USE_MAPPABLE == 0 {
            return Err(RutabagaError::InvalidBlob("not a mappable host3d blob"));
        }
        if info.size == 0 || info.size % BLOB_ALIGN != 0 {
            return Err(RutabagaError::InvalidBlob("size is not a multiple of 16 KiB"));
        }
        if self.blobs.contains_key(&resource_id) || self.plain.contains(&resource_id) {
            return Err(RutabagaError::InvalidResourceId);
        }
        // The executor finds blobs by (context, blob id), which the guest chooses: one per id.
        if self.blobs.values().any(|b| b.ctx_id == ctx_id && b.blob_id == info.blob_id) {
            return Err(RutabagaError::InvalidBlob("blob id already in use in this context"));
        }
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                info.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(RutabagaError::InvalidBlob("host mmap failed"));
        }
        assert_eq!(ptr as u64 % BLOB_ALIGN, 0, "the host page is 16 KiB");
        self.blobs.insert(resource_id, Blob { ctx_id, blob_id: info.blob_id, ptr, size: info.size });
        if let Some(o) = ops() {
            if let Some(f) = o.blob_created {
                f(o.ctx, ctx_id, info.blob_id, ptr, info.size);
            }
        }
        Ok(())
    }

    pub fn unref_resource(&mut self, resource_id: u32) -> RutabagaResult<()> {
        if self.plain.remove(&resource_id) {
            return Ok(());
        }
        let blob = self.blobs.remove(&resource_id).ok_or(RutabagaError::InvalidResourceId)?;
        // Metal may still read a blob it wraps after the guest frees it: the executor unmaps it when it's done.
        match ops().and_then(|o| o.blob_destroyed.map(|f| (o.ctx, f))) {
            Some((ctx, f)) => f(ctx, blob.ctx_id, blob.blob_id, blob.ptr, blob.size),
            None => unsafe {
                libc::munmap(blob.ptr, blob.size as usize);
            },
        }
        Ok(())
    }

    pub fn map_info(&self, resource_id: u32) -> RutabagaResult<u32> {
        self.blobs.get(&resource_id).ok_or(RutabagaError::InvalidResourceId)?;
        Ok(RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW)
    }

    pub fn map(&mut self, resource_id: u32) -> RutabagaResult<RutabagaMapping> {
        let blob = self.blobs.get(&resource_id).ok_or(RutabagaError::InvalidResourceId)?;
        Ok(RutabagaMapping { ptr: blob.ptr as u64, size: blob.size })
    }

    pub fn unmap(&mut self, resource_id: u32) -> RutabagaResult<()> {
        self.blobs.get(&resource_id).ok_or(RutabagaError::InvalidResourceId)?;
        Ok(())
    }

    pub fn export_blob(&mut self, resource_id: u32) -> RutabagaResult<RutabagaHandle> {
        self.blobs.get(&resource_id).ok_or(RutabagaError::InvalidResourceId)?;
        Ok(RutabagaHandle { handle_type: RUTABAGA_HANDLE_TYPE_MEM_DMABUF })
    }

    pub fn resource3d_info(&self, _resource_id: u32) -> RutabagaResult<Resource3DInfo> {
        Err(RutabagaError::Unsupported)
    }

    pub fn attach_backing(&mut self, _resource_id: u32, _iovecs: Vec<RutabagaIovec>) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn detach_backing(&mut self, _resource_id: u32) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn transfer_read(
        &mut self,
        _ctx_id: u32,
        _resource_id: u32,
        _transfer: Transfer3D,
        _buf: Option<IoSliceMut>,
    ) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn transfer_write(
        &mut self,
        _ctx_id: u32,
        _resource_id: u32,
        _transfer: Transfer3D,
        _buf: Option<IoSlice>,
    ) -> RutabagaResult<()> {
        Ok(())
    }

    pub fn resource_flush(&mut self, _resource_id: u32) -> RutabagaResult<()> {
        Err(RutabagaError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    // The fence handler and the ops are process-wide (one gpu device per VM host).
    static LOCK: Mutex<()> = Mutex::new(());

    fn rutabaga(seen: Arc<Mutex<Vec<RutabagaFence>>>) -> Rutabaga {
        let handler = RutabagaFenceHandler::new(move |f: RutabagaFence| seen.lock().unwrap().push(f));
        RutabagaBuilder::new(0, handler)
            .set_default_component(RutabagaComponentType::VirglRenderer)
            .build()
            .unwrap()
    }

    fn host3d(blob_id: u64, size: u64) -> ResourceCreateBlob {
        ResourceCreateBlob {
            blob_mem: VIRTIO_GPU_BLOB_MEM_HOST3D,
            blob_flags: VIRTIO_GPU_BLOB_FLAG_USE_MAPPABLE,
            blob_id,
            size,
        }
    }

    #[test]
    fn capset_is_dxmt_only() {
        let _l = LOCK.lock().unwrap();
        let r = rutabaga(Default::default());
        assert_eq!(r.get_capset_info(0).unwrap(), (MN_CAPSET_DXMT, 0, 0));
        assert!(r.get_capset_info(1).is_err());
    }

    #[test]
    fn context_rejects_other_capsets() {
        let _l = LOCK.lock().unwrap();
        let mut r = rutabaga(Default::default());
        assert!(r.create_context(1, 1, None).is_err());
        assert!(r.create_context(2, MN_CAPSET_DXMT, None).is_ok());
    }

    #[test]
    fn blob_size_must_be_16k_multiple() {
        let _l = LOCK.lock().unwrap();
        let mut r = rutabaga(Default::default());
        r.create_context(1, MN_CAPSET_DXMT, None).unwrap();
        assert!(r.resource_create_blob(1, 10, host3d(1, 16384), None, None).is_ok());
        assert!(r.resource_create_blob(1, 11, host3d(2, 4096), None, None).is_err());
    }

    #[test]
    fn duplicate_blob_id_refused() {
        let _l = LOCK.lock().unwrap();
        let mut r = rutabaga(Default::default());
        r.create_context(1, MN_CAPSET_DXMT, None).unwrap();
        r.create_context(2, MN_CAPSET_DXMT, None).unwrap();
        r.resource_create_blob(1, 10, host3d(5, 16384), None, None).unwrap();
        assert!(r.resource_create_blob(1, 11, host3d(5, 16384), None, None).is_err());
        assert!(r.resource_create_blob(2, 12, host3d(5, 16384), None, None).is_ok(), "blob ids are per context");
    }

    #[test]
    fn blob_map_returns_aligned_writable_ptr() {
        let _l = LOCK.lock().unwrap();
        let mut r = rutabaga(Default::default());
        r.create_context(1, MN_CAPSET_DXMT, None).unwrap();
        r.resource_create_blob(1, 10, host3d(1, 3 * 16384), None, None).unwrap();
        let m = r.map(10).unwrap();
        assert_eq!(m.ptr % 16384, 0);
        assert_eq!(m.size, 3 * 16384);
        unsafe {
            *((m.ptr + 16384 * 2 + 5) as *mut u8) = 0xa5;
            assert_eq!(*((m.ptr + 16384 * 2 + 5) as *const u8), 0xa5);
        }
        assert_eq!(r.map_info(10).unwrap() & RUTABAGA_MAP_ACCESS_MASK, RUTABAGA_MAP_ACCESS_RW);
        let handle = r.export_blob(10).unwrap();
        assert_eq!(handle.as_mesa_handle().unwrap().handle_type, RUTABAGA_HANDLE_TYPE_MEM_DMABUF);
        r.unref_resource(10).unwrap();
    }

    static mut DESTROYED: (u64, u64) = (0, 0);
    extern "C" fn take_blob(_ctx: *mut c_void, _ctx_id: u32, _blob_id: u64, ptr: *mut c_void, size: u64) {
        unsafe { DESTROYED = (ptr as u64, size) };
    }

    fn mapped(ptr: u64, size: u64) -> bool {
        unsafe { libc::msync(ptr as *mut c_void, size as usize, libc::MS_ASYNC) == 0 }
    }

    #[test]
    fn unref_hands_the_mapping_to_the_executor() {
        let _l = LOCK.lock().unwrap();
        let ops = MnDxmtOps {
            ctx: std::ptr::null_mut(),
            context_create: None,
            context_destroy: None,
            blob_created: None,
            blob_destroyed: Some(take_blob),
            submit: None,
            fence: None,
        };
        unsafe { mn_dxmt_set_ops(&ops) };
        let mut r = rutabaga(Default::default());
        r.create_context(1, MN_CAPSET_DXMT, None).unwrap();
        r.resource_create_blob(1, 10, host3d(1, 16384), None, None).unwrap();
        let m = r.map(10).unwrap();
        r.unref_resource(10).unwrap();
        let (ptr, size) = unsafe { DESTROYED };
        assert_eq!((ptr, size), (m.ptr, 16384));
        assert!(mapped(ptr, size), "the executor unmaps it, once nothing uses it");
        unsafe { libc::munmap(ptr as *mut c_void, size as usize) };
        unsafe { mn_dxmt_set_ops(std::ptr::null()) };
    }

    #[test]
    fn unref_without_an_executor_unmaps() {
        let _l = LOCK.lock().unwrap();
        unsafe { mn_dxmt_set_ops(std::ptr::null()) };
        let mut r = rutabaga(Default::default());
        r.create_context(1, MN_CAPSET_DXMT, None).unwrap();
        r.resource_create_blob(1, 10, host3d(1, 16384), None, None).unwrap();
        let m = r.map(10).unwrap();
        r.unref_resource(10).unwrap();
        assert!(!mapped(m.ptr, 16384));
    }

    extern "C" fn keep_fence(ctx: *mut c_void, _ctx_id: u32, _ring_idx: u32, _fence_id: u64) {
        unsafe { *(ctx as *mut u32) += 1 };
    }

    #[test]
    fn fence_done_reaches_handler() {
        let _l = LOCK.lock().unwrap();
        let mut pending: u32 = 0;
        let ops = MnDxmtOps {
            ctx: &mut pending as *mut u32 as *mut c_void,
            context_create: None,
            context_destroy: None,
            blob_created: None,
            blob_destroyed: None,
            submit: None,
            fence: Some(keep_fence),
        };
        unsafe { mn_dxmt_set_ops(&ops) };
        let seen: Arc<Mutex<Vec<RutabagaFence>>> = Default::default();
        let mut r = rutabaga(seen.clone());
        let fence = RutabagaFence { flags: 1 | 2, fence_id: 7, ctx_id: 1, ring_idx: 2 };
        r.create_fence(fence).unwrap();
        assert_eq!(pending, 1);
        assert!(seen.lock().unwrap().is_empty(), "the executor completes it, not create_fence");
        mn_dxmt_fence_done(1, 2, 7);
        let got = seen.lock().unwrap().clone();
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].flags, got[0].fence_id, got[0].ctx_id, got[0].ring_idx), (3, 7, 1, 2));
        unsafe { mn_dxmt_set_ops(std::ptr::null()) };
    }
}
