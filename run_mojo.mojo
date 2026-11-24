from memory.unsafe import DTypePointer
from vk_device import VulkanDeviceContext


fn main() raises:
    alias N: Int = 1024
    alias WORKGROUP_SIZE: Int = 64
    alias GRID_X: UInt32 = UInt32(N // WORKGROUP_SIZE)

    var ctx = VulkanDeviceContext()
    var q = ctx.create_queue()
    print("[Mojo] Pipeline cache bytes before load:", ctx.pipeline_cache_size())

    var bundle = ctx.load_bundle("kernel.spv")
    var saxpy = bundle.load_function("saxpy", 3)
    print("[Mojo] Pipeline cache bytes after load: ", ctx.pipeline_cache_size())

    let h_a = DTypePointer[DType.float32].alloc(N)
    let h_b = DTypePointer[DType.float32].alloc(N)
    let h_out = DTypePointer[DType.float32].alloc(N)
    for i in range(N):
        h_a.store(i, Float32(i))
        h_b.store(i, Float32(i) * 0.5)

    var d_a = ctx.create_buffer_f32(N)
    var d_b = ctx.create_buffer_f32(N)
    var d_c = ctx.create_buffer_f32(N)
    var d_d = ctx.create_buffer_f32(N)
    d_a.copy_from_host(h_a)
    d_b.copy_from_host(h_b)

    # Enqueue both dispatches into one batched command buffer (auto compute barrier)
    q.enqueue_3buf(saxpy, d_a, d_b, d_c, GRID_X)
    q.enqueue_3buf(saxpy, d_c, d_b, d_d, GRID_X)
    q.synchronize()

    d_d.copy_to_host(h_out)

    var max_err: Float32 = 0.0
    for i in range(N):
        let expected = Float32(2.5) * (Float32(2.5) * h_a.load(i) + h_b.load(i)) + h_b.load(i)
        var diff = h_out.load(i) - expected
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff
    print("[Mojo] Batched 2-pass chained SAXPY verified! max_err =", max_err)

    d_d.destroy()
    d_c.destroy()
    d_b.destroy()
    d_a.destroy()
    saxpy.destroy()
    bundle.destroy()
    q.destroy()
    ctx.destroy()
    h_a.free()
    h_b.free()
    h_out.free()
