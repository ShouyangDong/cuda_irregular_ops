from .loop_transformation import loop_recovery
from .loop_transformation import loop_split
from .loop_transformation import loop_fusion
from .loop_transformation import looop_reorder
from .Tensorization import detensorize, tensorize

def transform(code, source_platform, target_platform):
    """The transcompiler mainly consist of three process. 
    1. Loop transformation
    2. Memory conversion
    3. Tensorization
    Then we can check the code with correcsponding platform and
    transform by chain rule.
    """ 
    # Loop transformation
    if source_platform in ["NVIDIA GPU", "AMD MI", "Cambricon MLU"]:
        # we need to recovery the parallel variables into for loops
        code = loop_recovery(code)

    # we need to detensorize the intrinsic within the program
    if source_platform in ["Intel DL Boost", "Cambricon MLU"]:
        if source_platform == "Intel DL Boost":
            code = detensorize(code, intel_doc)
        else:
            code = detensorize(code, mlu_doc)

    # Memory conversion
    code = cache_read(code)
    code = cache_write(code)
    # Tensorization
    # Tensorize the intrinsic within the program
    if target_platform in ["Intel DL Boost", "Cambricon MLU"]:
        if source_platform == "Intel DL boost":
            code = tensorize(code, intel_doc)
        else:
            code = tensorize(code, mlu_doc)
    return code

if __name__ ==  "__main__":
    code = ""
    source_platform = ""
    target_platform = ""
