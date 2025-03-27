def generate_prior_from_src(code, src_target, dst_target):
  logit_prior = []
  # if src thread in code, then the loop recovery is high

  # if src instruction in the code, then the detensorization is high

  # if target thread  not in the code, then the auto_bind is high

  # if auto cache not in the code but needed, then the auto_cache is high

  # if target instruction not in the code, then auto_tensorization is high.
  
  return logit_prior
