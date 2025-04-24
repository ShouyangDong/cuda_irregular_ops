import argparse
import os
import sys

import openai

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.stderr.write(
        "Error: please set the OPENAI_API_KEY environment variable\n"
    )
    sys.exit(1)
openai.api_key = api_key

# Map (source_platform, dest_platform) to prompt templates
PROMPT_MAP = {
    ("cuda", "cpu"): CUDA_TO_CPU_PROMPT,
    ("cuda", "hip"): CUDA_TO_AMD_PROMPT,
    ("cuda", "mlu"): CUDA_TO_MLU_PROMPT,
    ("hip", "cpu"): HIP_TO_CPU_PROMPT,
    ("hip", "cuda"): HIP_TO_CUDA_PROMPT,
    ("hip", "mlu"): HIP_TO_MLU_PROMPT,
    ("cpu", "cuda"): CPU_TO_CUDA_PROMPT,
    ("cpu", "hip"): CPU_TO_HIP_PROMPT,
    ("cpu", "mlu"): CPU_TO_MLU_PROMPT,
    ("mlu", "cpu"): MLU_TO_CPU_PROMPT,
    ("mlu", "cuda"): MLU_TO_CUDA_PROMPT,
    ("mlu", "hip"): MLU_TO_HIP_PROMPT,
}


def code_transform(input_code: str, prompt_template: str) -> str:
    """Call the OpenAI o1 endpoint to perform zero-shot code translation."""
    prompt = prompt_template.format(input_code=input_code)
    response = openai.ChatCompletion.create(
        model="gpt-4o1",  # use the ‘o1’ optimized model for zero-shot
        messages=[
            {
                "role": "system",
                "content": "You are a code generation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot code translation between CUDA/CPU/AMD/MLU using o1"
    )
    parser.add_argument(
        "source_file", help="Path to the source code file to translate"
    )
    parser.add_argument(
        "source_platform",
        choices=["cuda", "hip", "cpu", "mlu"],
        help="Source platform (cuda | hip | cpu | mlu)",
    )
    parser.add_argument(
        "dest_platform",
        choices=["cuda", "hip", "cpu", "mlu"],
        help="Destination platform (cuda | hip | cpu | mlu)",
    )
    args = parser.parse_args()

    direction = (args.source_platform, args.dest_platform)
    if direction not in PROMPT_MAP:
        print(
            f"Error: unsupported translation direction {direction}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.source_file, "r", encoding="utf-8") as f:
        input_code = f.read()

    prompt_template = PROMPT_MAP[direction]
    translated_code = code_transform(input_code, prompt_template)

    output_dir = f"{args.source_platform}_{args.dest_platform}"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(args.source_file)
    output_path = os.path.join(output_dir, base_name)

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(translated_code)

    print(f"Translated code saved to: {output_path}")


if __name__ == "__main__":
    main()
