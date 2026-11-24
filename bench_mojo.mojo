from memory.unsafe import DTypePointer
from time import now
from vk_device import VulkanDeviceContext


fn bench_saxpy(ctx: VulkanDeviceContext) raises:
    alias N: Int = 4 * 1024 * 1024
    alias WG_SIZE: Int = 256
    alias GRID_X: UInt32 = UInt32(N // WG_SIZE)
    alias WARMUP: Int = 10
    alias ITERS: Int = 100

    var q = ctx.create_queue()

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

    let bytes_per_iter = Float64(N) * 4.0 * 6.0

    # 1. MLIR SPIR-V (batched via VulkanQueue)
    var b_mlir = ctx.load_bundle("bench_mlir.spv")
    var f_mlir = b_mlir.load_function("saxpy", 3)
    for _ in range(WARMUP):
        q.enqueue_3buf(f_mlir, d_a, d_b, d_c, GRID_X)
        q.enqueue_3buf(f_mlir, d_c, d_b, d_d, GRID_X)
        q.synchronize()

    let t0 = now()
    for _ in range(ITERS):
        q.enqueue_3buf(f_mlir, d_a, d_b, d_c, GRID_X)
        q.enqueue_3buf(f_mlir, d_c, d_b, d_d, GRID_X)
        q.synchronize()
    let t1 = now()
    let mlir_ms = Float64(t1 - t0) / 1e6 / Float64(ITERS)
    let mlir_gbps = (bytes_per_iter / 1e9) / (mlir_ms / 1e3)
    print("[Mojo Queue] SAXPY 4M MLIR GPU->SPIR-V:", mlir_ms, "ms/iter |", mlir_gbps, "GB/s")

    # 2. GLSL SPIR-V (batched via VulkanQueue)
    var b_glsl = ctx.load_bundle("bench_glsl.spv")
    var f_glsl = b_glsl.load_function("main", 3)
    for _ in range(WARMUP):
        q.enqueue_3buf(f_glsl, d_a, d_b, d_c, GRID_X)
        q.enqueue_3buf(f_glsl, d_c, d_b, d_d, GRID_X)
        q.synchronize()

    let t2 = now()
    for _ in range(ITERS):
        q.enqueue_3buf(f_glsl, d_a, d_b, d_c, GRID_X)
        q.enqueue_3buf(f_glsl, d_c, d_b, d_d, GRID_X)
        q.synchronize()
    let t3 = now()
    let glsl_ms = Float64(t3 - t2) / 1e6 / Float64(ITERS)
    let glsl_gbps = (bytes_per_iter / 1e9) / (glsl_ms / 1e3)
    print("[Mojo Queue] SAXPY 4M GLSL (glslc -O) :", glsl_ms, "ms/iter |", glsl_gbps, "GB/s")

    d_d.copy_to_host(h_out)
    var max_err: Float32 = 0.0
    for i in range(N):
        let expected = Float32(2.5) * (Float32(2.5) * h_a.load(i) + h_b.load(i)) + h_b.load(i)
        var diff = h_out.load(i) - expected
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff
    print("[Mojo Queue] SAXPY 4M verification    : max_err =", max_err)

    f_glsl.destroy()
    b_glsl.destroy()
    f_mlir.destroy()
    b_mlir.destroy()
    d_d.destroy()
    d_c.destroy()
    d_b.destroy()
    d_a.destroy()
    q.destroy()
    h_a.free()
    h_b.free()
    h_out.free()


fn bench_matmul(ctx: VulkanDeviceContext) raises:
    alias M: Int = 512
    alias ELEMS: Int = M * M
    alias GRID: UInt32 = UInt32(M // 16)
    alias WARMUP: Int = 10
    alias ITERS: Int = 100

    var q = ctx.create_queue()

    let h_a = DTypePointer[DType.float32].alloc(ELEMS)
    let h_b = DTypePointer[DType.float32].alloc(ELEMS)
    let h_c = DTypePointer[DType.float32].alloc(ELEMS)

    for r in range(M):
        for c in range(M):
            let idx = r * M + c
            h_a.store(idx, Float32(((r + c) % 17) - 8) * 0.125)
            h_b.store(idx, Float32(((r * 3 + c) % 13) - 6) * 0.125)
            h_c.store(idx, 0.0)

    var d_a = ctx.create_buffer_f32(ELEMS)
    var d_b = ctx.create_buffer_f32(ELEMS)
    var d_c = ctx.create_buffer_f32(ELEMS)
    d_a.copy_from_host(h_a)
    d_b.copy_from_host(h_b)

    let flops_per_iter = 2.0 * Float64(M) * Float64(M) * Float64(M)

    # 1. MLIR 2D Tiled MatMul (Workgroup LDS + gpu.barrier)
    var b_mlir = ctx.load_bundle("matmul_mlir.spv")
    var f_mlir = b_mlir.load_function("matmul_tiled", 3)
    for _ in range(WARMUP):
        q.enqueue_3buf(f_mlir, d_a, d_b, d_c, GRID, GRID, 1)
        q.synchronize()

    let t0 = now()
    for _ in range(ITERS):
        q.enqueue_3buf(f_mlir, d_a, d_b, d_c, GRID, GRID, 1)
        q.synchronize()
    let t1 = now()
    let mlir_ms = Float64(t1 - t0) / 1e6 / Float64(ITERS)
    let mlir_gflops = (flops_per_iter / 1e9) / (mlir_ms / 1e3)
    print("[Mojo Queue] MatMul 512x512 MLIR LDS  :", mlir_ms, "ms/iter |", mlir_gflops, "GFLOP/s")

    # Verify MLIR MatMul on sample rows (first 16 rows = 8,192 elements for fast CPU check)
    d_c.copy_to_host(h_c)
    var max_err_mlir: Float32 = 0.0
    for r in range(16):
        for c in range(M):
            var ref_val: Float32 = 0.0
            for k in range(M):
                ref_val += h_a.load(r * M + k) * h_b.load(k * M + c)
            var diff = h_c.load(r * M + c) - ref_val
            if diff < 0:
                diff = -diff
            if diff > max_err_mlir:
                max_err_mlir = diff
    print("[Mojo Queue] MatMul MLIR verification : max_err =", max_err_mlir)

    # 2. GLSL 2D Tiled MatMul (shared memory + barrier)
    var b_glsl = ctx.load_bundle("matmul_glsl.spv")
    var f_glsl = b_glsl.load_function("main", 3)
    for _ in range(WARMUP):
        q.enqueue_3buf(f_glsl, d_a, d_b, d_c, GRID, GRID, 1)
        q.synchronize()

    let t2 = now()
    for _ in range(ITERS):
        q.enqueue_3buf(f_glsl, d_a, d_b, d_c, GRID, GRID, 1)
        q.synchronize()
    let t3 = now()
    let glsl_ms = Float64(t3 - t2) / 1e6 / Float64(ITERS)
    let glsl_gflops = (flops_per_iter / 1e9) / (glsl_ms / 1e3)
    print("[Mojo Queue] MatMul 512x512 GLSL LDS  :", glsl_ms, "ms/iter |", glsl_gflops, "GFLOP/s")

    d_c.copy_to_host(h_c)
    var max_err_glsl: Float32 = 0.0
    for r in range(16):
        for c in range(M):
            var ref_val: Float32 = 0.0
            for k in range(M):
                ref_val += h_a.load(r * M + k) * h_b.load(k * M + c)
            var diff = h_c.load(r * M + c) - ref_val
            if diff < 0:
                diff = -diff
            if diff > max_err_glsl:
                max_err_glsl = diff
    print("[Mojo Queue] MatMul GLSL verification : max_err =", max_err_glsl)

    f_glsl.destroy()
    b_glsl.destroy()
    f_mlir.destroy()
    b_mlir.destroy()
    d_c.destroy()
    d_b.destroy()
    d_a.destroy()
    q.destroy()
    h_a.free()
    h_b.free()
    h_c.free()


fn main() raises:
    var ctx = VulkanDeviceContext()
    bench_saxpy(ctx)
    bench_matmul(ctx)
    print("[Mojo Queue] Final VkPipelineCache size (bytes):", ctx.pipeline_cache_size())
    ctx.destroy()
