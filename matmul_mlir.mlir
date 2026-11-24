// MLIR GPU-dialect 512x512 Tiled MatMul (16x16 tiles, Workgroup LDS + gpu.barrier)
// C[row, col] = sum_k A[row, k] * B[k, col]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    api=Vulkan,
    #spirv.resource_limits<
      max_compute_workgroup_invocations = 256,
      max_compute_workgroup_size = [256, 256, 64]
    >
  >
} {
  gpu.module @matmul_module {
    gpu.func @matmul_tiled(
      %A: memref<512x512xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>},
      %B: memref<512x512xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>},
      %C: memref<512x512xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}
    )
    kernel attributes {
      spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 16, 1]>
    } {
      %tileA = memref.alloc() : memref<16x16xf32, #spirv.storage_class<Workgroup>>
      %tileB = memref.alloc() : memref<16x16xf32, #spirv.storage_class<Workgroup>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      %zero = arith.constant 0.0 : f32

      %bx = gpu.block_id x
      %by = gpu.block_id y
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y

      %row_base = arith.muli %by, %c16 : index
      %row = arith.addi %row_base, %ty : index
      %col_base = arith.muli %bx, %c16 : index
      %col = arith.addi %col_base, %tx : index

      %acc_final = scf.for %t = %c0 to %c512 step %c16 iter_args(%acc = %zero) -> (f32) {
        %a_col = arith.addi %t, %tx : index
        %a_val = memref.load %A[%row, %a_col] : memref<512x512xf32>
        memref.store %a_val, %tileA[%ty, %tx] : memref<16x16xf32, #spirv.storage_class<Workgroup>>

        %b_row = arith.addi %t, %ty : index
        %b_val = memref.load %B[%b_row, %col] : memref<512x512xf32>
        memref.store %b_val, %tileB[%ty, %tx] : memref<16x16xf32, #spirv.storage_class<Workgroup>>

        gpu.barrier

        %acc_tile = scf.for %k = %c0 to %c16 step %c1 iter_args(%inner = %acc) -> (f32) {
          %la = memref.load %tileA[%ty, %k] : memref<16x16xf32, #spirv.storage_class<Workgroup>>
          %lb = memref.load %tileB[%k, %tx] : memref<16x16xf32, #spirv.storage_class<Workgroup>>
          %prod = arith.mulf %la, %lb : f32
          %sum = arith.addf %inner, %prod : f32
          scf.yield %sum : f32
        }

        gpu.barrier
        scf.yield %acc_tile : f32
      }

      memref.store %acc_final, %C[%row, %col] : memref<512x512xf32>
      gpu.return
    }
  }
}
