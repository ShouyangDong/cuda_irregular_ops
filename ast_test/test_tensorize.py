from pycparser import c_parser, c_ast, c_generator



if __name__ == "__main__":
    code = """
    void __bang_add(C, A, B, size) {
        #pragma operation(add)
        for (int i_add = 0; i_add < size; i_add++) {
            C[i_add] = A[i_add] + B[i_add];
        }
    }"
    """



    code = """
    void __bang_add(C, A, B, size) {
        #pragma __bang_add(input[Nram, Nram], output[Nram])
        for (int i_add = 0; i_add < size; i_add++) {
            C[i_add] = A[i_add] + B [i_add];
        }
    }