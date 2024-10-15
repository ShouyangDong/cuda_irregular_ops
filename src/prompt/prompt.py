SYSTEM_PROMPT = """
You are an expert performance engineer with deep experience in optimizing numerical linear algebra kernels for high-performance computing systems. 
Your primary goal is to optimize tensor operations to achieve peak efficiency on Deep Learning Processors (DLPs). 
You leverage techniques such as loop unrolling, SIMD vectorization, memory locality optimization, cache management, 
and hardware-specific parallelization strategies like tiling and pipelining.
"""


# apply optimization to the stage code according to the opt list
APPLY_OPT_PROMPT = \
"""
In code transformation, code optimization can be achieved by adding various types of compilation directive (pragmas). 
The following code is one stage of the whole algorithm: 
{STAGE_CODE_CONTENT}
Please apply the following optimization pragma to the above code: 
{OPT_LIST}
The parameter descriptions and usage examples of these pragmas are as follows: 
{PRAGMA_DEMO}
Please apply these pragmas in appropriate places based on parameter descriptions and reference examples.
Note that you only need to return the optimized code without any explaination. 
"""