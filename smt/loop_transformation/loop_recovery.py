import openai

question_system_prompt = """You are an expert compilation, and here is a code transformation task. 
                            You will generate the corresponding code based on the hints provided.
                            You should only output the C function without any explanation and natural language. 
                            Wrap your code with "```"
                            """


def prompt_generate(code, user_mannual):
    """
    Generate a prompt based on the presence of specific built-in variables in the code.

    This function checks if the provided code contains certain built-in variables that are
    indicative of parallel programming constructs. Depending on the presence of these
    variables, it generates a boolean prompt that can be used to guide the transformation
    of these variables into corresponding loop constructs.

    Parameters:
    - code (str): The source code to analyze.
    - user_mannual (str): Additional user manual or instructions (currently not used).

    Returns:
    - bool: A prompt indicating whether 'threadIdx' or 'coreId' is present in the code.
      - True if 'threadIdx' is found, suggesting a transformation to a thread-level loop.
      - False if 'coreId' is found, suggesting a transformation to a core-level loop.

    Raises:
    - ValueError: If both 'threadIdx' and 'coreId' are found in the code, which is
      ambiguous for transformation purposes.

    Todo:
    - Implement logic to handle cases where both 'threadIdx' and 'coreId' are present.
    - Extend the function to recognize more built-in variables and generate appropriate prompts.
    """
    # Check for the presence of 'threadIdx' and 'coreId' in the code
    has_threadIdx = "threadIdx" in code
    has_coreId = "coreId" in code

    # If both are present, raise an exception as it's ambiguous which to transform
    if has_threadIdx and has_coreId:
        raise ValueError("Ambiguous code: contains both 'threadIdx' and 'coreId'.")

    # Generate the prompt based on the presence of the built-in variables
    prompt = has_threadIdx

    return prompt


def gpt_transform(code, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


def transform_block(code, user_mannual):
    """
    Apply a series of transformations to the input code based on user manual instructions.

    This function first transforms the code using an AI model (presumably 'gpt') that
    understands the user manual instructions. It then tests the transformed code with a
    unittest to check for any issues. If the unittest indicates a failure, the code
    is further refined using a Satisfiability Modulo Theories (SMT) solver to fix any
    potential problems.

    Parameters:
    - code (str): The original source code to be transformed.
    - user_mannual (str): Instructions provided by the user to guide the transformation.

    Returns:
    - str: The transformed and potentially fixed source code.

    Raises:
    - NotImplementedError: If any of the transformation functions are not implemented.

    Todo:
    - Add error handling for cases where transformation or testing fails.
    - Improve the unittest to cover more edge cases.
    """
    # First transform the code using gpt
    prompt = prompt_generate(user_mannual)
    code = gpt_transform(code, prompt)
    # Test the code with unittest
    status = unittest(code)
    # Fix the code with SMT
    if not status:
        code = smt_transform(code)
    return code
