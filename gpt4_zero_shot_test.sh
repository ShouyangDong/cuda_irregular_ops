#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="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"  # <- replace with your OpenAI API key or export OPENAI_API_KEY

TRANSLATOR_PY="benchmark/zero_shot/code_transfer_gpt4.py"
BENCH_DIR="benchmark/data"

# 只跑 MLU→CPU 和 CPU→MLU
DIRECTIONS=(
  "mlu:cpu"
  "cpu:mlu"
)

# 编译脚本映射
declare -A COMPILE_SCRIPTS=(
  ["cpu"]="benchmark/evaluation/dlboost_test/compilation.py"
  ["mlu"]="benchmark/evaluation/mlu_test/compilation.py"
  ["cuda"]="benchmark/evaluation/cuda_test/compilation.py"
  ["hip"]="benchmark/evaluation/hip_test/compilation.py"
)


for dir_pair in "${DIRECTIONS[@]}"; do
  src_plat=${dir_pair%%:*}
  dst_plat=${dir_pair##*:}

  src_dir="$BENCH_DIR/${src_plat}_code_test"
  out_dir="translated/${src_plat}_to_${dst_plat}"

  echo
  echo "=== Pipeline: $src_plat → $dst_plat ==="

  # 1) 翻译
  mkdir -p "$out_dir"
  echo "-> Translating from $src_plat to $dst_plat into $out_dir"

  for src_file in "$src_dir"/*; do
    filename=$(basename "$src_file")
    dst_file="$out_dir/$filename"

    echo "   Translating $filename ..."
    python3 "$TRANSLATOR_PY" "$src_file" "$src_plat" "$dst_plat" > "$dst_file"
  done

  # 2) 编译
  compile_py="${COMPILE_SCRIPTS[$dst_plat]}"
  echo "-> Compiling translated code for $dst_plat using $compile_py"
  python3 "$compile_py" "$out_dir"

  # 3) 功能测试
  echo "-> Running correctness tests on $dst_plat with result_test.py"
  python3 "benchmark/evaluation/${dst_plat}_test/result_test.py" "$out_dir" "benchmark/evaluation/${dst_plat}_test"
done

echo
echo "=== All Done ==="
