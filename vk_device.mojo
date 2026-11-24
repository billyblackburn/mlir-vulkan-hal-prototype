from sys.ffi import DLHandle
from memory.unsafe import DTypePointer, Pointer

alias OpaquePtr = DTypePointer[DType.uint8]


struct VulkanBuffer:
    var handle: OpaquePtr
    var size_bytes: UInt64
    var _lib: DLHandle

    fn __init__(inout self, lib: DLHandle, handle: OpaquePtr, size_bytes: UInt64):
        self._lib = lib
        self.handle = handle
        self.size_bytes = size_bytes

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self.handle = existing.handle
        self.size_bytes = existing.size_bytes
        existing.handle = OpaquePtr()

    fn device_address(self) -> UInt64:
        let addr_fn = self._lib.get_function[fn (OpaquePtr) -> UInt64](
            "vk_hal_buffer_device_address"
        )
        return addr_fn(self.handle)

    fn copy_from_host(self, src: DTypePointer[DType.float32]) raises:
        let upload_fn = self._lib.get_function[
            fn (OpaquePtr, DTypePointer[DType.float32], UInt64) -> Int32
        ]("vk_hal_buffer_upload")
        let rc = upload_fn(self.handle, src, self.size_bytes)
        if rc != 0:
            raise Error("vk_hal_buffer_upload failed with code " + String(rc))

    fn copy_to_host(self, dst: DTypePointer[DType.float32]) raises:
        let download_fn = self._lib.get_function[
            fn (OpaquePtr, DTypePointer[DType.float32], UInt64) -> Int32
        ]("vk_hal_buffer_download")
        let rc = download_fn(self.handle, dst, self.size_bytes)
        if rc != 0:
            raise Error("vk_hal_buffer_download failed with code " + String(rc))

    fn destroy(inout self):
        if self.handle:
            let destroy_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_buffer_destroy"
            )
            destroy_fn(self.handle)
            self.handle = OpaquePtr()


struct VulkanEvent:
    var handle: OpaquePtr
    var _lib: DLHandle

    fn __init__(inout self, lib: DLHandle, handle: OpaquePtr):
        self._lib = lib
        self.handle = handle

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self.handle = existing.handle
        existing.handle = OpaquePtr()

    fn synchronize(self) raises:
        let sync_fn = self._lib.get_function[fn (OpaquePtr) -> Int32](
            "vk_hal_event_synchronize"
        )
        let rc = sync_fn(self.handle)
        if rc != 0:
            raise Error("vk_hal_event_synchronize failed with code " + String(rc))

    fn is_ready(self) -> Bool:
        let ready_fn = self._lib.get_function[fn (OpaquePtr) -> Int32](
            "vk_hal_event_is_ready"
        )
        return ready_fn(self.handle) != 0

    fn destroy(inout self):
        if self.handle:
            let destroy_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_event_destroy"
            )
            destroy_fn(self.handle)
            self.handle = OpaquePtr()


struct VulkanFunction:
    var handle: OpaquePtr
    var binding_count: UInt32
    var push_constant_size: UInt32
    var _lib: DLHandle

    fn __init__(
        inout self,
        lib: DLHandle,
        handle: OpaquePtr,
        binding_count: UInt32,
        push_constant_size: UInt32 = 0,
    ):
        self._lib = lib
        self.handle = handle
        self.binding_count = binding_count
        self.push_constant_size = push_constant_size

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self.handle = existing.handle
        self.binding_count = existing.binding_count
        self.push_constant_size = existing.push_constant_size
        existing.handle = OpaquePtr()

    fn destroy(inout self):
        if self.handle:
            let unload_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_function_unload"
            )
            unload_fn(self.handle)
            self.handle = OpaquePtr()


struct VulkanBundle:
    var handle: OpaquePtr
    var _lib: DLHandle

    fn __init__(inout self, lib: DLHandle, handle: OpaquePtr):
        self._lib = lib
        self.handle = handle

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self.handle = existing.handle
        existing.handle = OpaquePtr()

    fn load_function(
        self, entry_point: String, binding_count: UInt32
    ) raises -> VulkanFunction:
        let load_fn = self._lib.get_function[
            fn (OpaquePtr, DTypePointer[DType.int8], UInt32) -> OpaquePtr
        ]("vk_hal_function_load")
        let fn_h = load_fn(self.handle, entry_point._as_ptr(), binding_count)
        _ = entry_point
        if not fn_h:
            raise Error("vk_hal_function_load failed")
        return VulkanFunction(self._lib, fn_h, binding_count, 0)

    fn load_function_bda(
        self, entry_point: String, push_constant_size: UInt32
    ) raises -> VulkanFunction:
        let load_fn = self._lib.get_function[
            fn (OpaquePtr, DTypePointer[DType.int8], UInt32) -> OpaquePtr
        ]("vk_hal_function_load_bda")
        let fn_h = load_fn(
            self.handle, entry_point._as_ptr(), push_constant_size
        )
        _ = entry_point
        if not fn_h:
            raise Error("vk_hal_function_load_bda failed")
        return VulkanFunction(self._lib, fn_h, 0, push_constant_size)

    fn destroy(inout self):
        if self.handle:
            let unload_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_bundle_unload"
            )
            unload_fn(self.handle)
            self.handle = OpaquePtr()


struct VulkanQueue:
    var handle: OpaquePtr
    var _lib: DLHandle

    fn __init__(inout self, lib: DLHandle, handle: OpaquePtr):
        self._lib = lib
        self.handle = handle

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self.handle = existing.handle
        existing.handle = OpaquePtr()

    fn enqueue_3buf(
        self,
        func: VulkanFunction,
        b0: VulkanBuffer,
        b1: VulkanBuffer,
        b2: VulkanBuffer,
        groups_x: UInt32,
        groups_y: UInt32 = 1,
        groups_z: UInt32 = 1,
    ) raises:
        let ptrs = Pointer[OpaquePtr].alloc(3)
        ptrs.store(0, b0.handle)
        ptrs.store(1, b1.handle)
        ptrs.store(2, b2.handle)

        let exec_fn = self._lib.get_function[
            fn (
                OpaquePtr,
                OpaquePtr,
                Pointer[OpaquePtr],
                UInt32,
                UInt32,
                UInt32,
                UInt32,
            ) -> Int32
        ]("vk_hal_queue_execute")
        let rc = exec_fn(
            self.handle, func.handle, ptrs, UInt32(3), groups_x, groups_y, groups_z
        )
        ptrs.free()
        if rc != 0:
            raise Error("vk_hal_queue_execute failed with code " + String(rc))

    fn enqueue_bda(
        self,
        func: VulkanFunction,
        push_data: OpaquePtr,
        push_size: UInt32,
        groups_x: UInt32,
        groups_y: UInt32 = 1,
        groups_z: UInt32 = 1,
    ) raises:
        let exec_fn = self._lib.get_function[
            fn (
                OpaquePtr,
                OpaquePtr,
                OpaquePtr,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
            ) -> Int32
        ]("vk_hal_queue_execute_bda")
        let rc = exec_fn(
            self.handle, func.handle, push_data, push_size, groups_x, groups_y, groups_z
        )
        if rc != 0:
            raise Error("vk_hal_queue_execute_bda failed with code " + String(rc))

    fn record_event(self, ev: VulkanEvent) raises:
        let rec_fn = self._lib.get_function[fn (OpaquePtr, OpaquePtr) -> Int32](
            "vk_hal_queue_record_event"
        )
        let rc = rec_fn(self.handle, ev.handle)
        if rc != 0:
            raise Error("vk_hal_queue_record_event failed with code " + String(rc))

    fn wait_event(self, ev: VulkanEvent) raises:
        let wait_fn = self._lib.get_function[fn (OpaquePtr, OpaquePtr) -> Int32](
            "vk_hal_queue_wait_event"
        )
        let rc = wait_fn(self.handle, ev.handle)
        if rc != 0:
            raise Error("vk_hal_queue_wait_event failed with code " + String(rc))

    fn synchronize(self) raises:
        let sync_fn = self._lib.get_function[fn (OpaquePtr) -> Int32](
            "vk_hal_queue_synchronize"
        )
        let rc = sync_fn(self.handle)
        if rc != 0:
            raise Error("vk_hal_queue_synchronize failed with code " + String(rc))

    fn destroy(inout self):
        if self.handle:
            let destroy_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_queue_destroy"
            )
            destroy_fn(self.handle)
            self.handle = OpaquePtr()


struct VulkanDeviceContext:
    var handle: OpaquePtr
    var _lib: DLHandle

    fn __init__(inout self, device_index: UInt32 = 0) raises:
        self._lib = DLHandle("./libvk_hal.so")
        let create_fn = self._lib.get_function[fn (UInt32) -> OpaquePtr](
            "vk_hal_context_create"
        )
        self.handle = create_fn(device_index)
        if not self.handle:
            raise Error("Failed to initialize VulkanDeviceContext")

    fn pipeline_cache_size(self) -> UInt64:
        let cache_fn = self._lib.get_function[fn (OpaquePtr) -> UInt64](
            "vk_hal_context_pipeline_cache_size"
        )
        return cache_fn(self.handle)

    fn create_queue(self) raises -> VulkanQueue:
        let create_fn = self._lib.get_function[fn (OpaquePtr) -> OpaquePtr](
            "vk_hal_queue_create"
        )
        let q = create_fn(self.handle)
        if not q:
            raise Error("Failed to create VulkanQueue")
        return VulkanQueue(self._lib, q)

    fn create_event(self) raises -> VulkanEvent:
        let create_fn = self._lib.get_function[fn (OpaquePtr) -> OpaquePtr](
            "vk_hal_event_create"
        )
        let ev = create_fn(self.handle)
        if not ev:
            raise Error("Failed to create VulkanEvent")
        return VulkanEvent(self._lib, ev)

    fn create_buffer_f32(self, count: Int) raises -> VulkanBuffer:
        let size_bytes = UInt64(count * 4)
        let create_fn = self._lib.get_function[
            fn (OpaquePtr, UInt64) -> OpaquePtr
        ]("vk_hal_buffer_create")
        let buf = create_fn(self.handle, size_bytes)
        if not buf:
            raise Error("Failed to allocate device-local VulkanBuffer")
        return VulkanBuffer(self._lib, buf, size_bytes)

    fn load_bundle(self, spv_path: String) raises -> VulkanBundle:
        let load_fn = self._lib.get_function[
            fn (OpaquePtr, DTypePointer[DType.int8]) -> OpaquePtr
        ]("vk_hal_bundle_load")
        let b = load_fn(self.handle, spv_path._as_ptr())
        _ = spv_path
        if not b:
            raise Error("vk_hal_bundle_load failed")
        return VulkanBundle(self._lib, b)

    fn destroy(inout self):
        if self.handle:
            let destroy_fn = self._lib.get_function[fn (OpaquePtr) -> None](
                "vk_hal_context_destroy"
            )
            destroy_fn(self.handle)
            self.handle = OpaquePtr()
