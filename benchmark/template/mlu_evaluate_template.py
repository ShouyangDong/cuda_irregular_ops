import re
from string import Template


def create_bang_perf_func(file_name, op_type="ewise"):
    with open(file_name, "r") as f:
        original_function = f.read()
        f.close()

    # 正则表达式提取参数部分
    function_signature_pattern = r"void (\w+)\(([^()]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find function signature.")

    # 获取函数名称和参数列表
    kernel_name = match.group(1)
    param_list_str = match.group(2)

    # 构造参数列表
    params = [param_str.strip() for param_str in param_list_str.split(",")]
    param_list = ", ".join(
        [
            " ".join(param.split()[:-1]) + " " + param.split()[-1]
            for param in params
        ]
    )
    mlu_params = [param + "_mlu" for param in params]
    mlu_param_list = ", ".join(
        [
            " ".join(param.split()[:-1]) + " " + param.split()[-1]
            for param in mlu_params
        ]
    )

    dim = None
    func_type = None
    if "clusterId" in original_function:
        dim = "cnrtDim3_t dim = {16, 1, 1};"
        func_type = "cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_UNION4;"
    elif "coreId" in original_function:
        dim = "cnrtDim3_t dim = {4, 1, 1};"
        func_type = "cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_UNION1;"
    else:
        dim = "cnrtDim3_t dim = {1, 1, 1};"
        func_type = "cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;"
    # 构造新的计时函数模板
    device_memory_alloc = []
    memcpy = []
    size = None
    if op_type == "ewise":
        size = "size"
        for param in params:
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_mlu;\n")
            device_memory_alloc.append(
                f"CNRT_CHECK(cnrtMalloc((void**)&{name}_mlu, {size} * sizeof(float)));\n"
            )

        for param in params[:-1]:
            name = param.split("*")[1]
            memcpy.append(
                f"CNRT_CHECK(cnrtMemcpy({name}_mlu, {name}, {size} * sizeof(float), cnrtMemcpyHostToDev));\n"
            )
        # copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"CNRT_CHECK(cnrtMemcpy({name}, {name}_mlu, {size} * sizeof(float), cnrtMemcpyDevToHost));\n"
    elif op_type == "pool":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_mlu;\n")
            device_memory_alloc.append(
                f"CNRT_CHECK(cnrtMalloc((void**)&{name}_mlu, {size[i]} * sizeof(float)));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            memcpy.append(
                f"CNRT_CHECK(cnrtMemcpy({name}_mlu, {name}, {size[i]} * sizeof(float), cnrtMemcpyHostToDev));\n"
            )
        # copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"CNRT_CHECK(cnrtMemcpy({name}, {name}_mlu, {size[-1]} * sizeof(float), cnrtMemcpyDevToHost));\n"
    elif op_type == "matmul":
        size = ["size1", "size2", "size3"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_mlu;\n")
            device_memory_alloc.append(
                f"CNRT_CHECK(cnrtMalloc((void**)&{name}_mlu, {size[i]} * sizeof({dtype})));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"CNRT_CHECK(cnrtMemcpy({name}_mlu, {name}, {size[i]} * sizeof({dtype}), cnrtMemcpyHostToDev));\n"
            )
        # copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"CNRT_CHECK(cnrtMemcpy({name}, {name}_mlu, size3 * sizeof({dtype}), cnrtMemcpyDevToHost));\n"

    elif op_type == "layer_norm":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_mlu;\n")
            if i == 1 or i == 2:
                device_memory_alloc.append(
                    f"CNRT_CHECK(cnrtMalloc((void**)&{name}_mlu, size2 * sizeof(float)));\n"
                )
            else:
                device_memory_alloc.append(
                    f"CNRT_CHECK(cnrtMalloc((void**)&{name}_mlu, size1 * sizeof(float)));\n"
                )
        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            if i == 1 or i == 2:
                memcpy.append(
                    f"CNRT_CHECK(cnrtMemcpy({name}_mlu, {name}, size2 * sizeof(float), cnrtMemcpyHostToDev));\n"
                )
            else:
                memcpy.append(
                    f"CNRT_CHECK(cnrtMemcpy({name}_mlu, {name}, size1 * sizeof(float), cnrtMemcpyHostToDev));\n"
                )
        # copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"CNRT_CHECK(cnrtMemcpy({name}, {name}_mlu, size1 * sizeof(float), cnrtMemcpyDevToHost));\n"

    memory_free = []
    for param in params:
        name = param.split("*")[1]
        memory_free.append(f"cnrtFree({name}_mlu);\n")

    if isinstance(size, list):
        size_list = ", ".join(
            arg for arg in ["int " + string for string in size]
        )
    else:
        size_list = "int size"

    cpp_pef_template = Template(
        """
    #include <stdio.h>
    #include <bang.h>
    #include <stdint.h>

    // Original function
    ${original_function}

    extern "C" float timed_${kernel_name}_kernel(${param_list}, ${size_list}) {
        cnrtQueue_t queue;
        CNRT_CHECK(cnrtSetDevice(1));
        CNRT_CHECK(cnrtQueueCreate(&queue));
        ${dim}
        ${func_type}
        cnrtNotifier_t start, end;
        CNRT_CHECK(cnrtNotifierCreate(&start));
        CNRT_CHECK(cnrtNotifierCreate(&end));
        ${memcpy_alloc_list}
        ${memcpy_list}
        for (int i = 0; i < 10; i++) {
            ${kernel_name}<<<dim, ktype, queue>>>(${called_param_list});
        }
        CNRT_CHECK(cnrtPlaceNotifier(start, queue));
        for (int i = 0; i < 1000; i++) {
           ${kernel_name}<<<dim, ktype, queue>>>(${called_param_list});
        }
        CNRT_CHECK(cnrtPlaceNotifier(end, queue));
        cnrtQueueSync(queue);
        ${memcpy_back}
        float timeTotal;
        CNRT_CHECK(cnrtNotifierDuration(start, end, &timeTotal));
        float ms_time = timeTotal / 1000.0 / 1000.0;
        printf("Total Time: %.3f ms\\n", ms_time);
        CNRT_CHECK(cnrtQueueDestroy(queue));
        ${memory_free}
        return ms_time;
    }
    """
    )

    pattern = r'extern\s*"C"\s*'
    # 使用 re.sub 替换匹配部分为空字符串
    cleaned_code = re.sub(pattern, "", original_function)

    called_param_list = mlu_param_list.replace("float *", "")
    called_param_list = called_param_list.replace("int8 *", "")
    called_param_list = called_param_list.replace("int *", "")
    memcpy_alloc_list = "        ".join(alloc for alloc in device_memory_alloc)
    memcpy_list = "        ".join(cpy for cpy in memcpy)
    memory_free_list = "        ".join(cpy for cpy in memory_free)
    # 动态替换模板
    new_code = cpp_pef_template.substitute(
        kernel_name=kernel_name,
        param_list=param_list,
        called_param_list=called_param_list,
        original_function=cleaned_code,
        memcpy_alloc_list=memcpy_alloc_list,
        dim=dim,
        func_type=func_type,
        memcpy_list=memcpy_list,
        memcpy_back=memcpy_back,
        memory_free=memory_free_list,
        size_list=size_list,
    )

    # 保存生成的 C++ 文件
    output_file = file_name.replace(".mlu", "_bak.mlu")
    with open(output_file, "w") as f:
        f.write(new_code)
    return output_file


if __name__ == "__main__":
    file_name = "benchmark/data/mlu_code_test/layernorm_2_4_32.mlu"
    create_bang_perf_func(
        "benchmark/data/mlu_code_test/layernorm_2_4_32.mlu", "layer_norm"
    )
