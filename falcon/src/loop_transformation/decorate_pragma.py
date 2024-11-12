SPLIT_PRAGMA_PROMPT = """
Please analyze the following loop and determine which axes can be split using loop splitting. 
Additionally, provide the appropriate pragma directive for loop splitting above the for loop.


### Transformation:
1. **Identify Axes for Loop Split**: Assess the loop to determine if the iteration variable `i` can be split.

2. **Specify the Split Factor**: Choose an appropriate factor for splitting the loop. 
This factor should be a divisor of the loop's range. 
For example, if the factor is 4, the loop will be split into 4 segments.

3. **Add the Pragma Directive**: Insert the `#pragma` directive for loop splitting above the loop.

### Requirements:
- Refactoring code to add pragmas without modifying the existing logic or function names.

### Input Code:
Please provide the C++ code containing nested `for` loops suitable for loop reorder.

{code}

### Output Code:
Return the transformed C++ code after annotating the loop_split pragma.

"""

SPLIT_PRAGMA_DEMO = """
DEMO

### Original Code:
```cpp
void mul(float* A, float* B, float* C) {
    for (int i = 0; i < 60; i++) {
        A[i] = B[i] * C[i];
    }
}

```

### After Transformation:
```cpp
void mul(float* A, float* B, float* C) {
    #pragma loop_split(factor)
    for (int i = 0; i < 60; i++) {
        A[i] = B[i] * C[i];
    }
}
```


### Example Input:
```c
void add(float* A, float* B, float* C, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
```

### Example Output:
```c
void add(float* A, float* B, float* C, int N, int M) {
    for (int i = 0; i < N; i++) {
        #pragma loop_split(factor)
        for (int j = 0; j < M; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
```
"""