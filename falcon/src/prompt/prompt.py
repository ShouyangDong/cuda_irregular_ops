SYSTEM_PROMPT = """
You are an expert performance engineer with deep experience in optimizing numerical linear algebra kernels for high-performance computing systems.
Your primary goal is to optimize tensor operations to achieve peak efficiency on Deep Learning Processors (DLPs).
You leverage techniques such as loop unrolling, SIMD vectorization, memory locality optimization, cache management,
and hardware-specific parallelization strategies like tiling and pipelining.
"""


# select suitable opt option from the pragma list
PRAGMA_INSERT_PROMPT = """
In code transformation, pragmas are compiler directives that guide the optimization process.
Each pragma has specific scenarios and cannot be applied arbitrarily.
Here is the function description and the applicable scenarios for the pragma:
{PRAGMA_DESCRIPTION}
The following is one stage of the tensor program. Your task is to insert the pragma {PRAGMA_NAME}
**only** above the specific code block that it applies to.
Instructions:
- **Do not modify** the code in any other way.
- **Do not add pragmas** in inappropriate places or across the entire code.
- **Do not provide optimized code**, only insert the pragma above the corresponding code block.
- Ensure the pragma is correctly positioned to affect only the intended section of the code.
Here is the code where the pragma should be inserted:
{STAGE_CODE_CONTENT}
Please insert the pragma directly above the relevant code block, without changing the existing code content. """

# apply optimization to the stage code according to the opt list
APPLY_OPT_PROMPT = """
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
