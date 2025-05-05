#!/usr/bin/env bash
set -euo pipefail

TRANSLATOR_PY="benchmark/zero_shot/code_transfer_gpt4.py"
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

# 编译脚本映射
declare -A COMPILE_SCRIPTS=(
	["cpu"]="benchmark/evaluation/dlboost_test/compilation.py"
	["mlu"]="benchmark/evaluation/mlu_test/compilation.py"
	["cuda"]="benchmark/evaluation/cuda_test/compilation.py"
	["hip"]="benchmark/evaluation/hip_test/compilation.py"
)

# 测试脚本映射
declare -A TEST_SCRIPTS=(
	["cpu"]="benchmark/evaluation/dlboost_test/result_test.py"
	["mlu"]="benchmark/evaluation/mlu_test/result_test.py"
	["cuda"]="benchmark/evaluation/cuda_test/result_test.py"
	["hip"]="benchmark/evaluation/hip_test/result_test.py"
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

	files=("$src_dir"/*)
	total=${#files[@]}
	((total == 0)) && {
		echo "   [WARN] no files in $src_dir"
		exit 0
	}

	i=0
	for src_file in "${files[@]}"; do
		((i++))
		filename=$(basename "$src_file")
		dst_file="$out_dir/$filename"

		# 打印进度，格式：Translating 文件名 [当前/总数]
		printf "   Translating %-20s [%3d/%3d]\r" "$filename" "$i" "$total"

		python3 "$TRANSLATOR_PY" "$src_file" "$src_plat" "$dst_plat"
	done

	# 循环结束后换行，防止进度条残留
	printf "\n"

	# 2) 编译
	compile_py="${COMPILE_SCRIPTS[$dst_plat]}"
	echo "-> Compiling translated code for $dst_plat using $compile_py"
	python3 "$compile_py" "$out_dir"

	# 3) 功能测试
	test_py="${TEST_SCRIPTS[$dst_plat]}"
	echo "-> Running correctness tests on $dst_plat with result_test.py"
	python3 "$test_py" "$out_dir" "benchmark/evaluation/${dst_plat}_test"
done
echo "=== All Done ==="
