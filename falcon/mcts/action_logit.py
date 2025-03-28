import re


def generate_prior_from_src(code, src_target, dst_target):
    """
    根据源代码中出现的特定关键词，为各个转换 pass 分配优先级。

    参数：
      code: 字符串，源代码内容。
      src_target: 源平台类型（"cuda" 或 "bangc"）。
      dst_target: 目标平台类型（"cuda" 或 "bangc"）。

    返回：
      logit_prior: 包含 (action, priority) 元组的列表，其中 priority 为 "high" 或 "default"。
    """
    logit_prior = []

    # 定义各平台线程变量和指令模式
    cuda_paravar = [
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
    ]
    mlu_paravar = ["coreId", "clusterId"]

    # 1. 如果源代码中含有源平台的线程变量，则 loop_recovery 优先级设为 high
    if src_target.lower() == "cuda":
        for token in cuda_paravar:
            if token in code:
                logit_prior.append(0.5)

    elif src_target.lower() == "bangc":
        for token in mlu_paravar:
            if token in code:
                logit_prior.append(0.5)
    else:
        logit_prior.append(0.05)

    # 2. 如果源代码中包含源平台的指令，则 detensorization 优先级设为 high
    # 对于 bangc 平台，指令以 "bang" 开头；对于 cuda，可简单检测 "cuda" 关键词
    if src_target.lower() == "bangc":
        if re.search(r"\bbang\w*", code):
            logit_prior.append(0.5)

    elif src_target.lower() == "cuda":
        if "cuda" in code.lower():
            logit_prior.append(0.5)

    # 3. 如果源代码中不包含目标平台的线程变量，则 auto_bind 优先级设为 high
    if dst_target.lower() == "cuda":
        if not any(token in code for token in cuda_paravar):
            logit_prior.append(0.5)
    elif dst_target.lower() == "bangc":
        if not any(token in code for token in mlu_paravar):
            logit_prior.append(0.5)

    # 4. 如果代码中没有出现与缓存相关的内容（这里简单检测 "cache" 关键字），则 auto_cache 设为 high
    if "cache" not in code.lower():
        logit_prior.append(0.5)

    # 5. 如果代码中缺少目标平台的指令，则 auto_tensorization 设为 high
    # 对于 bangc 平台，指令以 "bang" 开头；对于 cuda，简单检测 "cuda" 关键词
    if dst_target.lower() == "bangc":
        if not re.search(r"\bbang\w*", code):
            logit_prior.append(0.5)
    elif dst_target.lower() == "cuda":
        if "cuda" not in code.lower():
            logit_prior.append(0.5)

    # 对于其他 pass（stmt_split, loop_fusion, loop_reorder, loop_split,
    # loop_contraction, auto_pipeline），暂时赋予默认优先级
    other_actions = [
        "stmt_split",
        "loop_fusion",
        "loop_reorder",
        "loop_split",
        "loop_contraction",
        "auto_pipeline",
    ]
    for action in other_actions:
        logit_prior.append(0.1)

    return logit_prior
