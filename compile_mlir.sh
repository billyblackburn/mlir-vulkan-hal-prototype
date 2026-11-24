#!/usr/bin/env bash
set -euo pipefail

# 1. Lower MLIR gpu.module -> spirv.module
pixi run mlir-opt kernel.mlir \
  --pass-pipeline='builtin.module(convert-gpu-to-spirv,spirv.module(spirv-lower-abi-attrs,spirv-update-vce,canonicalize,cse))' \
  -o kernel_spirv.mlir

# 2. Extract the spirv.module and serialize to binary SPIR-V
python3 -c '
text = open("kernel_spirv.mlir").read()
start = text.index("  spirv.module")
end = text.index("  gpu.module")
open("kernel_spirv_only.mlir", "w").write(text[start:end])
'
pixi run mlir-translate --no-implicit-module --serialize-spirv kernel_spirv_only.mlir -o kernel.spv

# 3. Validate SPIR-V against Vulkan 1.3
spirv-val --target-env vulkan1.3 kernel.spv
echo "Compiled and validated kernel.spv ($(stat -c%s kernel.spv) bytes)"
