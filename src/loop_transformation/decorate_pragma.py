SPLIT_PRAGMA_PROMPT = """
Please analyze the following loop and determine which axes can be split using loop splitting. 
Additionally, provide the appropriate pragma directive for loop splitting.

### Original Code:
```cpp
for (int i = 0; i < 60; i++) {
    A[i] = B[i] * C[i];
}
```

### Transformation:
1. **Identify Axes for Loop Split**: Assess the loop to determine if the iteration variable `i` can be split.

2. **Specify the Split Factor**: Choose an appropriate factor for splitting the loop. This factor should be a divisor of the loop's range. For example, if the factor is 4, the loop will be split into 4 segments.

3. **Add the Pragma Directive**: Insert the `#pragma` directive for loop splitting before the loop.

### After Transformation:
```cpp
#pragma loop_split(factor)
for (int i = 0; i < 60; i++) {
    A[i] = B[i] * C[i];
}
``

`### Input Code:
Please provide the C++ code containing nested `for` loops suitable for loop reorder.

{code}

### Output Code:
Return the transformed C++ code after annotating the loop_split pragma.
"""
