from memory.unsafe import DTypePointer
from time import now
from vk_device import OpaquePtr, VulkanDeviceContext


fn pack_saxpy_args(
    buf: OpaquePtr,
    ptr_a: UInt64,
    ptr_b: UInt64,
    ptr_c: UInt64,
    alpha: Float32,
    n: UInt32,
):
    # 32-byte push constant layout matching saxpy_bda.comp:
    #   [0..7]   uint64 A
    #   [8..15]  uint64 B
    #   [16..23] uint64 C
    #   [24..27] float  alpha
    #   [28..31] uint32 N
    buf.bitcast[DType.uint64]().store(0, ptr_a)
    buf.bitcast[DType.uint64]().store(1, ptr_b)
    buf.bitcast[DType.uint64]().store(2, ptr_c)
    buf.offset(24).bitcast[DType.float32]().store(0, alpha)
    buf.offset(28).bitcast[DType.uint32]().store(0, n)


fn main() raises:
    alias N: Int = 4 * 1024 * 1024
    alias WG_SIZE: Int = 256
    alias GRID_X: UInt32 = UInt32((N + WG_SIZE - 1) // WG_SIZE)
    alias PUSH_SIZE: UInt32 = 32
    alias WARMUP: Int = 10
    alias ITERS: Int = 100

    var ctx = VulkanDeviceContext()
    var q1 = ctx.create_queue()
    var q2 = ctx.create_queue()
    var ev = ctx.create_event()

    let h_a = DTypePointer[DType.float32].alloc(N)
    let h_b = DTypePointer[DType.float32].alloc(N)
    let h_out = DTypePointer[DType.float32].alloc(N)
    for i in range(N):
        let v = Float32(i & 1023)
        h_a.store(i, v)
        h_b.store(i, v * 0.5)

    var d_a = ctx.create_buffer_f32(N)
    var d_b = ctx.create_buffer_f32(N)
    var d_c = ctx.create_buffer_f32(N)
    var d_d = ctx.create_buffer_f32(N)
    d_a.copy_from_host(h_a)
    d_b.copy_from_host(h_b)

    let addr_a = d_a.device_address()
    let addr_b = d_b.device_address()
    let addr_c = d_c.device_address()
    let addr_d = d_d.device_address()
    print("[Mojo BDA] GPU 64-bit device addresses:")
    print("  d_a =", addr_a, "| d_b =", addr_b)
    print("  d_c =", addr_c, "| d_d =", addr_d)

    var bundle = ctx.load_bundle("saxpy_bda.spv")
    var fn_bda = bundle.load_function_bda("main", PUSH_SIZE)

    let args1 = OpaquePtr.alloc(32)
    let args2 = OpaquePtr.alloc(32)
    # Runtime scalar args! Pass 1 uses alpha = 2.5, Pass 2 uses alpha = 3.0
    pack_saxpy_args(args1, addr_a, addr_b, addr_c, Float32(2.5), UInt32(N))
    pack_saxpy_args(args2, addr_c, addr_b, addr_d, Float32(3.0), UInt32(N))

    # --- Test 1: Cross-Queue GPU-Side Timeline Semaphore Sync ---
    # q1 runs Pass 1 -> records timeline event (async, zero CPU wait!)
    # q2 waits on event on the GPU -> runs Pass 2 with different alpha -> sync
    q1.enqueue_bda(fn_bda, args1, PUSH_SIZE, GRID_X)
    q1.record_event(ev)
    q2.wait_event(ev)
    q2.enqueue_bda(fn_bda, args2, PUSH_SIZE, GRID_X)
    q2.synchronize()

    print("[Mojo Timeline] Cross-queue event ready after q2.synchronize():", ev.is_ready())

    d_d.copy_to_host(h_out)
    var max_err: Float32 = 0.0
    for i in range(N):
        let expected = Float32(3.0) * (Float32(2.5) * h_a.load(i) + h_b.load(i)) + h_b.load(i)
        var diff = h_out.load(i) - expected
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff
    print("[Mojo BDA+Timeline] Cross-queue 4M verification: max_err =", max_err)

    # --- Test 2: Single-Queue Batched BDA Benchmark (zero descriptor sets) ---
    let bytes_per_iter = Float64(N) * 4.0 * 6.0
    for _ in range(WARMUP):
        q1.enqueue_bda(fn_bda, args1, PUSH_SIZE, GRID_X)
        q1.enqueue_bda(fn_bda, args2, PUSH_SIZE, GRID_X)
        q1.synchronize()

    let t0 = now()
    for _ in range(ITERS):
        q1.enqueue_bda(fn_bda, args1, PUSH_SIZE, GRID_X)
        q1.enqueue_bda(fn_bda, args2, PUSH_SIZE, GRID_X)
        q1.synchronize()
    let t1 = now()
    let bda_ms = Float64(t1 - t0) / 1e6 / Float64(ITERS)
    let bda_gbps = (bytes_per_iter / 1e9) / (bda_ms / 1e3)
    print("[Mojo BDA Queue] SAXPY 4M (0 descriptors):", bda_ms, "ms/iter |", bda_gbps, "GB/s")

    args1.free()
    args2.free()
    fn_bda.destroy()
    bundle.destroy()
    ev.destroy()
    q2.destroy()
    q1.destroy()
    d_d.destroy()
    d_c.destroy()
    d_b.destroy()
    d_a.destroy()
    ctx.destroy()
    h_a.free()
    h_b.free()
    h_out.free()
