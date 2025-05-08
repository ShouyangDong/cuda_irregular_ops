#!/usr/bin/env bash
set -euo pipefail

TRANSLATOR_PY="falcon/mcts/transcompile_without_smt.py"
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

	echo
	echo "=== Pipeline: $src_plat → $dst_plat ==="
	echo "-> Translating from $src_plat to $dst_plat"

	files=("$src_dir"/*)
	total=${#files[@]}
	((total == 0)) && {
		echo "   [WARN] no files in $src_dir"
		continue
	}

	i=0
	for src_file in "$src_dir"/*; do
		((i+=1))
		filename=$(basename "$src_file")

		printf "   Translating %-20s [%3d/%3d]\r" \
			"$filename" "$i" "$total"
		python3 "$TRANSLATOR_PY" \
			--source "$src_plat" \
			--target "$dst_plat" \
			--file_name "$src_file"
	done
	printf "\n"
done
echo "=== All Done ==="
