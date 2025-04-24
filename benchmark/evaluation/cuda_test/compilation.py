#!/usr/bin/env python3
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_cuda_compilation as run_compilation


def compile_file(file_name):
    name_no_ext, _ = os.path.splitext(os.path.basename(file_name))

    # 1) 读取原始代码 + 加入宏
    with open(file_name, "r") as f:
        code = f.read()
    macro_path = os.path.join("benchmark", "macro", "cuda_macro.txt")
    with open(macro_path, "r") as f:
        macro = f.read()
    full_code = macro + code

    # 2) 写入备份 .cu
    bak_file = os.path.join(
        os.path.dirname(file_name), f"{name_no_ext}_bak.cu"
    )
    with open(bak_file, "w") as f:
        f.write(full_code)

    # 3) 构造 .so 路径（与 bak_file 在同一目录）
    so_path = os.path.join(os.path.dirname(bak_file), f"{name_no_ext}.so")

    # 4) 编译
    success, output = run_compilation(so_path, bak_file)

    # 5) 清理
    os.remove(bak_file)
    if success and os.path.exists(so_path):
        os.remove(so_path)

    if not success:
        print(f"[ERROR] Failed to compile {file_name}:\n{output}", flush=True)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Batch-compile CUDA .cu files in a given directory"
    )
    parser.add_argument(
        "src_dir",
        help="Directory containing .cu files to compile (e.g. translated/nvidia_cpu)",
    )
    args = parser.parse_args()

    # 从命令行参数获取目录，然后查找所有 .cu
    pattern = os.path.join(args.src_dir, "*.cu")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No .cu files found in {args.src_dir}", file=sys.stderr)
        return

    # 并行编译并统计
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_file, files), total=len(files))
        )

    total = len(files)
    succ = sum(results)
    print(
        f"[INFO] Compilation success rate: {succ}/{total} = {succ/total:.2%}"
    )


if __name__ == "__main__":
    main()
