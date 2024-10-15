import re
import time
import openai

from prompt.prompt import (
    SYSTEM_PROMPT,
    TASK_PIPELINE_FOR_SINGLE_FUNCTION_PROMPT,
    TASK_PIPELINE_PROMPT,
    TASK_PIPELINE_STRATEGY_PROMPT_4,
)


OPT_LIST = [
    "THREAD_BINDING",
    "CACHE_READ",
    "CACHE_WRITE",
    "TENSORIZATION",
    "DOUBLE_BUFFER",
]

openai.api_key = """ OPENAI API KEY """

# split the whole code
def gen_task_pipeline_prompt(func_description):
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


# split sub function in the code
def gen_sub_task_pipeline_prompt(func_description, func_content):
    _SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{ALGO_NAME}", algo_name)
    _SYSTEM_PROMPT = _SYSTEM_PROMPT.replace("{FUNCTION_DESCRIPTION}", func_description)

    TASK_PIPELINE_PROMPT_COMPLETE = (
        _SYSTEM_PROMPT + TASK_PIPELINE_FOR_SINGLE_FUNCTION_PROMPT + func_content
    )
    TASK_PIPELINE_PROMPT_COMPLETE += (
        TASK_PIPELINE_STRATEGY_PROMPT_4  # add strategy hint
    )
    print(TASK_PIPELINE_PROMPT_COMPLETE)
    return TASK_PIPELINE_PROMPT_COMPLETE


def extract_content_and_save(
    chat_completion, prompt_content, model_name, sub_func=False
):
    if model_name in ["gpt-4-1106-preview", "gpt-4"]:
        model_name = "gpt4"
    else:
        model_name = "gpt3.5"
    cur_time = time.strftime("%y%m%d_%H%M", time.localtime())
    chat_file_path = f"{pipeline_log_path}/{model_name}/{algo_name}/pipeline_{model_name}_{algo_name}_{cur_time}.txt"
    code_file_path = f"{pipeline_log_path}/{model_name}/{algo_name}/pipeline_{model_name}_{algo_name}_{cur_time}.cpp"

    if sub_func:
        chat_file_path = f"{pipeline_log_path}/{model_name}/{algo_name}/sub_pipeline_{model_name}_{algo_name}_{cur_time}.txt"
        code_file_path = f"{pipeline_log_path}/{model_name}/{algo_name}/sub_pipeline_{model_name}_{algo_name}_{cur_time}.cpp"

    # save chat completion
    with open(chat_file_path, "w") as chat_file:
        chat_file.write(str(chat_completion))
        chat_file.write("\n\n====================================\n\n")
        chat_file.write(prompt_content)

    # extract code from ``` ``` and save
    content = chat_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        print(code_content)
        with open(code_file_path, "w") as code_file:
            code_file.write(code_content)


def run_task_pipeline(func_description):
    prompt = gen_task_pipeline_prompt(func_description)
    model_name = "gpt-4-1106-preview"
    task_pipeline_completion = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    print(task_pipeline_completion)
    extract_content_and_save(task_pipeline_completion, prompt, model_name, algo_name)


def run_sub_task_pipeline(func_description, func_content):
    prompt = gen_sub_task_pipeline_prompt(func_description, func_content)
    model_name = "gpt-4-1106-preview"
    sub_task_pipeline_completion = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    print(sub_task_pipeline_completion)
    extract_content_and_save(
        sub_task_pipeline_completion, prompt, model_name, sub_func=True
    )


def connection_test():
    prompt = "hello"
    model_name = "gpt-4-1106-preview"
    test_completion = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    print(test_completion)


if __name__ == "__main__":
    func_content = """
    extern "C" void add_kernel(float* output, float* input1, float* input2) {
        int dim1 = 4;
        int dim2 = 4;
        int dim3 = 4;
        int dim4 = 64;
        
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
                        output[index] = input1[index] + input2[index];
                    }
                }
            }
        }
    }
    """
    run_task_pipeline(func_content)
