#!/usr/bin/env bash
# 批量编译 cuda/cu/*.cu -> cuda/so/*.so
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CU_DIR="$SCRIPT_DIR/cu"
OUT_DIR="$SCRIPT_DIR/so"

mkdir -p "$OUT_DIR"

for f in "$CU_DIR"/*.cu; do
    name="$(basename "$f" .cu)"
    echo "[nvcc] $name.cu -> $name.so"
    nvcc -shared -Xcompiler -fPIC "$f" -o "$OUT_DIR/$name.so"
done

echo
echo "All CUDA operators compiled to $OUT_DIR"
