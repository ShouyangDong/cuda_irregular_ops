from falcon.mcts.actions import actions as ActionSpace


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
    logit_prior = [0.2] * len(ActionSpace)

    if src_target == "cuda" and "thread" in code:
        logit_prior[0] = 0.5

    if src_target == "mlu" and "coreId" in code:
        logit_prior[0] = 0.5

    if src_target == "hip" and "thread" in code:
        logit_prior[0] = 0.5

    if src_target == "cuda" and "mma_sync" in code:
        logit_prior[2] = 0.5

    if src_target == "hip" and "amdgcn" in code:
        logit_prior[2] = 0.5

    if src_target == "mlu" and "__bang" in code:
        logit_prior[2] = 0.5

    if src_target == "cpu" and "dpbusd" in code:
        logit_prior[2] = 0.5

    if dst_target == "cuda" or dst_target == "hip" and "thread" not in code:
        logit_prior[7] = 0.4

    if dst_target == "mlu" and "coreId" not in code:
        logit_prior[7] = 0.4

    if dst_target == "mlu" and "__bang" not in code:
        logit_prior[9] = 0.4

    return logit_prior
