import re

import openai

from falcon.simplification import simplify_code
from falcon.src.loop_transformation.decorate_pragma import SPLIT_PRAGMA_PROMPT
from falcon.src.loop_transformation.pass_prompt import (
    LOOP_FUSION_DEMO,
    LOOP_FUSION_PROMPT,
    LOOP_REORDER_DEMO,
    LOOP_REORDER_PROMPT,
    LOOP_SPLIT_DEMO,
    LOOP_SPLIT_PROMPT,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def run_loop_fusion(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_FUSION_PROMPT}
    {LOOP_FUSION_DEMO}
    Please return the output kernel function without any additional information.
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_FUSION_PROMPT}", LOOP_FUSION_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_FUSION_DEMO}", LOOP_FUSION_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return simplify_code(code_content)
    return None


def run_loop_reorder(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_REORDER_PROMPT}
    {LOOP_REORDER_DEMO}
    Please return the output kernel function without any additional information.
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_REORDER_PROMPT}", LOOP_REORDER_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_REORDER_DEMO}", LOOP_REORDER_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def run_split_annotation(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {SPLIT_PRAGMA_PROMPT}
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{SPLIT_PRAGMA_PROMPT}", SPLIT_PRAGMA_PROMPT)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def run_apply_split(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_SPLIT_PROMPT}
    {LOOP_SPLIT_DEMO}
    Please return the output kernel function without any additional information.
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_SPLIT_PROMPT}", LOOP_SPLIT_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_SPLIT_DEMO}", LOOP_SPLIT_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None
