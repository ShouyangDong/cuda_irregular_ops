#!/usr/bin/env bash
set -euo pipefail

# Config
export OPENAI_API_KEY="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"  # <- replace with your OpenAI API key or export OPENAI_API_KEY
TRANSLATOR_PY="benchmark/zero_shot/code_transfer_gpt4.py"  # <- path to your Python translation script
BENCH_DIR="benchmark/data"
PLATFORMS=(cuda cpu hip mlu)

# Helper: map directory name to platform key
# cuda_code_test  -> cuda
# dlboost_code_test -> cpu
# hip_code_test   -> hip
# mlu_code_test   -> mlu
declare -A DIR2PLATFORM=(
  ["cuda_code_test"]="cuda"
  ["dlboost_code_test"]="cpu"
  ["hip_code_test"]="hip"
  ["mlu_code_test"]="mlu"
)

COMPILE_SCRIPTS=(
  [cuda]=benchmark/evaluation/cuda_test/compilation.py
  [hip]=benchmark/evaluation/hip_test/compilation.py
  [cpu]=benchmark/evaluation/dlboost_test/compilation.py
  [mlu]=benchmark/evaluation/mlu_test/compilation.py
)

# Output root
OUT_ROOT="translated"
mkdir -p "${OUT_ROOT}"

echo "=== Starting zero-shot translation + compile + test ==="

for dir in "${!DIR2PLATFORM[@]}"; do
  src_plat=${DIR2PLATFORM[$dir]}
  src_dir="${BENCH_DIR}/${dir}"
  echo ">> Processing directory ${src_dir} (source platform=${src_plat})"

  for src_file in "${src_dir}"/*.cpp; do
    fname=$(basename "$src_file")
    name_no_ext="${fname%.cpp}"

    for dst_plat in "${PLATFORMS[@]}"; do
      # skip identity
      if [[ "$dst_plat" == "$src_plat" ]]; then
        continue
      fi

      echo "  - Translating $fname from $src_plat → $dst_plat ..."

      # Prepare output directory
      out_dir="${OUT_ROOT}/${src_plat}_to_${dst_plat}"
      mkdir -p "$out_dir"
      out_file="${out_dir}/${fname}"

      # Run translator (zero-shot)
      python3 "$TRANSLATOR_PY" \
        "$src_file" "$src_plat" "$dst_plat" \
        > "$out_file"

      echo "    -> written translated code to $out_file"

      # Compile the translated code
      for src_dir in translated/*_*; do
        direction=$(basename "$src_dir")
        dst_plat=${direction#*_to_}   # “nvidia_to_cpu” → “cpu”
        compile_script=${COMPILE_SCRIPTS[$dst_plat]:-}

        if [[ -z "$compile_script" ]]; then
          echo "No compile script for target platform '$dst_plat', skipping."
          continue
        fi

        echo "=== Compiling all translations in '$src_dir' for platform '$dst_plat' ==="
        if ! python3 "$compile_script" "$src_dir"; then
          echo ">>> ERROR: compilation script '$compile_script' failed for '$src_dir'"
          exit 1
        fi
        echo
      done

      # Run performance test
      echo "    -> running perf test ..."
      perf_out="${out_dir}/${name_no_ext}_${dst_plat}_perf.txt"
      case "$dst_plat" in
        cuda)
          CUDA_VISIBLE_DEVICES=0 "$bin" --perf > "$perf_out"
          ;;
        hip)
          HIP_VISIBLE_DEVICES=0 "$bin" --perf > "$perf_out"
          ;;
        cpu|mlu)
          "$bin" --perf > "$perf_out"
          ;;
      esac
      echo "       saved perf results to $perf_out"

      # Run accuracy test
      echo "    -> running accuracy test ..."
      acc_out="${out_dir}/${name_no_ext}_${dst_plat}_acc.txt"
      case "$dst_plat" in
        cuda)
          CUDA_VISIBLE_DEVICES=0 "$bin" --accuracy > "$acc_out"
          ;;
        hip)
          HIP_VISIBLE_DEVICES=0 "$bin" --accuracy > "$acc_out"
          ;;
        cpu|mlu)
          "$bin" --accuracy > "$acc_out"
          ;;
      esac
      echo "       saved accuracy results to $acc_out"
    done
  done
done

echo "=== All done. Results under $OUT_ROOT ==="
