import openai

opt_options = [
    "LOOP_FUSION",
    "LOOP_REORDER",
    "LOOP_SPLIT",
    "LOOP_RECOVERY",
    "THREAD_BINDING",
    "TENSOR_COMTRACTION",
    "CACHE_READ",
    "CACHE_WRITE",
    "TENSORIZATION",
    "DETENSORIZATION",
    "DOUBLE_BUFFER",
]


model_name = "gpt-4-turbo"


def gen_stage_opt_prompt(stage_code):
    STAGE_CODE_CONTENT = stage_code
    PRAGMA_DESCRIPTION = ""
    for opt in opt_options:
        prompt_name = f"{opt}_PROMPT"
        prompt_content = globals()[prompt_name]
        PRAGMA_DESCRIPTION += prompt_content
        PRAGMA_DESCRIPTION += "-----------------------"
    _OPT_CHOICE_PROMPT = OPT_CHOICE_PROMPT.replace(
        "{PRAGMA_DESCRIPTION}", PRAGMA_DESCRIPTION
    )
    _OPT_CHOICE_PROMPT = _OPT_CHOICE_PROMPT.replace(
        "{STAGE_CODE_CONTENT}", STAGE_CODE_CONTENT
    )
    print(_OPT_CHOICE_PROMPT)
    return _OPT_CHOICE_PROMPT


# completion_type: opt_choose or opt_apply
def save_chat_completion(completion_type, prompt_content, chat_completion, model_name):
    experiment_name = completion_type
    if model_name in ["gpt-4-1106-preview", "gpt-4"]:
        model_name = "gpt4"
    else:
        model_name = "gpt3.5"

    cur_time = time.strftime("%y%m%d_%H%M", time.localtime())
    chat_file_path = f"{root_path}/log/stage_opt/{model_name}/{experiment_name}_{model_name}_{cur_time}.txt"
    with open(chat_file_path, "w") as chat_file:
        chat_file.write(str(chat_completion))
        chat_file.write("\n\n====================================\n\n")
        chat_file.write(prompt_content)

    # extract code from ``` ``` and save
    if completion_type == "opt_apply":
        code_file_path = f"{root_path}/log/stage_opt/{model_name}/{experiment_name}_{model_name}_{cur_time}.cpp"
        content = chat_completion.choices[0].message["content"]
        match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
        if match:
            code_content = match.group(1)
            print(code_content)
            with open(code_file_path, "w") as code_file:
                code_file.write(code_content)


def parse_opt_list(stage_opt_completion):
    raw_list_str = stage_opt_completion.choices[0].message["content"]
    raw_list_str = raw_list_str[1:-1]
    opt_list = raw_list_str.split(", ")
    return opt_list


def apply_opt(stage_code, stage_opt_list, algo_name, func_description):
    _SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{ALGO_NAME}", algo_name)
    _SYSTEM_PROMPT = _SYSTEM_PROMPT.replace("{FUNCTION_DESCRIPTION}", func_description)

    STAGE_CODE_CONTENT = stage_code
    OPT_LIST = ""
    PRAGMA_DEMO_COMPLETE = ""

    opt_code = "xxx"
    for i, opt_option in enumerate(stage_opt_list):
        OPT_LIST += str(i) + ". " + opt_option + "\n"
        pragma_name = opt_option.split(" ")[-1].upper() + "_DEMO"
        PRAGMA_DEMO = globals()[pragma_name]
        PRAGMA_DEMO_COMPLETE += str(i) + ". " + opt_option + ":\n" + PRAGMA_DEMO + "\n"

    _APPLY_OPT_PROMPT = APPLY_OPT_PROMPT.replace(
        "{STAGE_CODE_CONTENT}", STAGE_CODE_CONTENT
    )
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{OPT_LIST}", OPT_LIST)
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{PRAGMA_DEMO}", PRAGMA_DEMO_COMPLETE)

    STAGE_OPT_PROMPT_COMPLETE = _SYSTEM_PROMPT + _APPLY_OPT_PROMPT
    print(STAGE_OPT_PROMPT_COMPLETE)

    opt_apply_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": STAGE_OPT_PROMPT_COMPLETE}],
    )
    print(opt_apply_completion)
    save_chat_completion(
        "opt_apply", STAGE_OPT_PROMPT_COMPLETE, opt_apply_completion, model_name
    )
    return opt_code


"""
model_list = ["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-1106", "gpt-3.5-turbo"]
"""


def run_stage_code_opt(stage_code):
    prompt = gen_stage_opt_prompt(stage_code)
    stage_opt_completion = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    print(stage_opt_completion)
    stage_opt_list = parse_opt_list(stage_opt_completion)
    save_chat_completion("opt_choose", prompt, stage_opt_completion, model_name)
    opt_code = apply_opt(stage_code, stage_opt_list)
    return opt_code


if __name__ == "__main__":
    stage_code = """
    __global__ void __launch_bounds__(1024) add(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
            if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + ((int)blockIdx.x)) < 2048000) {
            T_add[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (A[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + B[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))]);
            }
        }
    }
    """
    run_stage_code_opt(stage_code)
