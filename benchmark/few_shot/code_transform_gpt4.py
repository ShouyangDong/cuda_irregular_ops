import openai

openai.api_key = "your-api-key"  # 替换为你的 OpenAI API Key

CUDA_TO_CPU_PROMPT_TEMPLATE = """
You are an expert in low-level deep learning compiler optimization.

Task:
Translate the following CUDA kernel into optimized CPU code using AVX VNNI intrinsics if possible.

Constraints:
- Match the numerical accuracy.
- Try to preserve parallelism using SIMD (e.g., AVX VNNI).
- Use intrinsics instead of OpenMP or naive loops.
- Keep code complete with includes, main function, and comments.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent optimized CPU code:
"""


def code_transform(input_code, source_platform, dst_platform):
    prompt = CUDA_TO_CPU_PROMPT_TEMPLATE.format(input_code=input_code)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a code generation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    result = response["choices"][0]["message"]["content"]
    return result
