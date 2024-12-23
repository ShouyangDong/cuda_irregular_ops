import logging
import os
import sys

import bangpy as bp
import numpy as np
from bangpy import autotuning, tcp

# TARGET = enabled_targets()
TARGET = "mlu590-h8"
#########################################################################
# Define the Search Space
# -------------------------------
# In this section, we will build a sufficient space to cover the techniques used
# in these tutorials and then rely on the efficient auto-tuner to search
# through this space and pick the best configurations.
# If you are familiar with writing CAMBRICON BANGPy TCP, you can find the following
# template is very general. This template is easily modified
# to tune other operators such as gemm.
# To fully understand this template, you should be familiar with TCP
# API and auto-tuning API.
# Note that the search space for a matmul operator can be
# very large for some input shapes.


# sample code
@autotuning.template("tutorial/matmul")
def matmul(N, L, M):
    tcps = tcp.TCP(TARGET)
    data = tcps.Buffer(
        shape=(N, L), name="data", dtype=bp.int16, scope="global"
    )
    filters = tcps.Buffer(
        shape=(L, M), name="filter", dtype=bp.int16, scope="global"
    )
    output = tcps.Buffer(
        shape=(N, M), name="output", dtype=bp.float32, scope="global"
    )

    ##### define space begin ######
    cfg = autotuning.get_config()
    cfg.define_split("tile_N", N, num_outputs=2)
    cfg.define_split("tile_M", M // 64, num_outputs=2)
    ##### define space end ######

    # Implement kernel according to configuration.
    N_o, N_i = cfg["tile_N"].apply(N)
    M_o, M_i = cfg["tile_M"].apply(M // 64)
    data_block = tcps.Buffer(
        shape=(1, L), name="date_block", dtype=bp.int16, scope="nram"
    )
    with tcps.for_range(0, N_o) as no:
        with tcps.for_range(0, N_i) as ni:
            tcps.memcpy(
                data_block.reshape(
                    [
                        L,
                    ]
                ),
                data[no * N_i + ni],
            )

            with tcps.for_range(0, M_o) as m:
                filter_block = tcps.Buffer(
                    shape=(L, 64 * M_i),
                    name="filter_block",
                    dtype=bp.int16,
                    scope="wram",
                )
                output_block = tcps.Buffer(
                    shape=(1, 64 * M_i),
                    name="output_block",
                    dtype=bp.float32,
                    scope="nram",
                )
                tcps.memcpy(
                    filter_block, filters[:, m * 64 * M_i : (m + 1) * 64 * M_i]
                )
                tcps.dense(output_block, data_block, filter_block, 0)
                tcps.memcpy(
                    output[no * N_i + ni, m * 64 * M_i : (m + 1) * 64 * M_i],
                    output_block.reshape(
                        [
                            64 * M_i,
                        ]
                    ),
                )
    cfg.add_flop(N * L * M * 2)
    f_matmul = tcps.BuildBANG(
        inputs=[data, filters], outputs=[output], kernel_name="matmul"
    )
    return f_matmul, [data, filters, output]


#######################################################################
# Search Through the Space
# --------------------------------

# Logging configuration (for printing tuning log to screen).
logging.getLogger("autotuning").setLevel(logging.DEBUG)
logging.getLogger("autotuning").addHandler(logging.StreamHandler(sys.stdout))

N, L, M = 512, 512, 512
task = autotuning.task.create("tutorial/matmul", args=(N, L, M), target=TARGET)
print(task.config_space)

# Use local MLU, measure 10 times for every configuration to reduce variance.
# The timeout for compiling a program is 10 seconds, the timeout for
# running is 4 seconds.
measure_option = autotuning.measure_option(
    builder=autotuning.LocalBuilder(),
    runner=autotuning.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file matmul.log.
# During tuning we will also try many invalid configurations, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotuning.RandomTuner(task)
tuner.tune(
    n_trial=20,
    measure_option=measure_option,
    callbacks=[autotuning.callback.log_to_file("matmul.log")],
)

############################################################################
# Finally, you can inspect the best configuration from the log file, check correctness,
# and measure the spent time.

# Inspect the best configuration.
dispatch_context = autotuning.apply_history_best("matmul.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# Apply history best from log file.
with autotuning.apply_history_best("matmul.log"):
    f_matmul, _ = matmul(N, L, M)

# Check correctness.
a_np = np.random.uniform(low=0, high=1, size=(N, L))
w_np = np.random.uniform(low=0, high=1, size=(L, M))
c_np = np.random.uniform(low=0, high=1, size=(N, M))
dev = bp.device(0)
a_bp = bp.Array(a_np.astype(np.int16), dev)
w_bp = bp.Array(w_np.astype(np.int16), dev)
c_bp = bp.Array(c_np.astype(np.float32), dev)
f_matmul(a_bp, w_bp, c_bp)
bp.assert_allclose(
    c_bp.numpy(),
    np.matmul(a_np.astype("int16"), w_np.astype("int16")).astype("float32"),
    rtol=1e-2,
    atol=1e-2,
)

evaluator = f_matmul.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
print(
    "Time cost of this operator : %f ms"
    % (evaluator(a_bp, w_bp, c_bp).mean * 1e3)
)
os.remove("matmul.log")
