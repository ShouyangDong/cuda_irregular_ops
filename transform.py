

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
        code = loop_recovert(code)

    if source_platform in ["Intel DL Boost", "Cambricon MLU"]:
        # we need to detensorize the intrinsic within the program

    # Memory conversion


    # Tensorization

    if target_platform in ["Intel DL Boost", "Cambricon MLU"]:
        # Tensorize the intrinsic within the program

    return code

if __name__ ==  "__main__":
    code = ""
    source_platform = ""
    target_platform = ""
