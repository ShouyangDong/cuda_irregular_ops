import openai


def run_task_pipeline(algo_name, func_description):
    prompt = gen_task_pipeline_prompt(algo_name, func_description)
    model_name = "gpt-4-1106-preview"
    task_pipeline_completion = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    print(task_pipeline_completion)
    extract_content_and_save(task_pipeline_completion, prompt, model_name, algo_name)


def gen_task_pipeline_prompt(algo_name, func_description):
    _SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{ALGO_NAME}", algo_name)
    _SYSTEM_PROMPT = _SYSTEM_PROMPT.replace("{FUNCTION_DESCRIPTION}", func_description)

    code_path = f"{benchmark_dir}/{algo_name}/{algo_name}.cpp"
    with open(code_path, "r") as code_file:
        CODE = "\n" + code_file.read()

    TASK_PIPELINE_PROMPT_COMPLETE = _SYSTEM_PROMPT + TASK_PIPELINE_PROMPT + CODE
    TASK_PIPELINE_PROMPT_COMPLETE += (
        TASK_PIPELINE_STRATEGY_PROMPT_4  # add strategy hint
    )
    print(TASK_PIPELINE_PROMPT_COMPLETE)
    return TASK_PIPELINE_PROMPT_COMPLETE
