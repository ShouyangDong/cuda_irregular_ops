#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="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"  # <- replace with your OpenAI API key or export OPENAI_API_KEY

TRANSLATOR_PY="benchmark/few_shot/code_transfer_gpt4.py"
BENCH_DIR="benchmark/data"

# 只跑 MLU→CPU 和 CPU→MLU
DIRECTIONS=(
  "mlu:cpu"
  "cpu:mlu"
  "mlu:hip"
  "mlu:cuda"
  "cpu:hip"
  "cpu:cuda"
  "cuda:mlu"
  "cuda:hip"
  "cuda:cpu"
  "hip:mlu"
  "hip:cuda"
  "hip:cpu"
)

for dir_pair in "${DIRECTIONS[@]}"; do
  src_plat=${dir_pair%%:*}
  dst_plat=${dir_pair##*:}

  src_dir="$BENCH_DIR/${src_plat}_code_test"
  out_dir="${src_plat}_${dst_plat}"

  echo
  echo "=== Pipeline: $src_plat → $dst_plat ==="

  # 1) 翻译
  mkdir -p "$out_dir"
  echo "-> Translating from $src_plat to $dst_plat into $out_dir"

  for src_file in "$src_dir"/*; do
    filename=$(basename "$src_file")
    dst_file="$out_dir/$filename"

    echo "   Translating $filename ..."
    python3 "$TRANSLATOR_PY" "$src_file" "$src_plat" "$dst_plat"
  done
done
echo "=== All Done ==="
