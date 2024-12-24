import os
import tempfile
from functools import partial

import jax as jx
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from benchmark.perf import perf_bang, perf_dlboost
from falcon.mcts.actions import actions as ActionSpace

GFLOPS = 64 * 1280 * 2 / 1e9
A_Length = len(ActionSpace)


def objective(file_name, target):
    """We design an objective function. If compile and runtime error happens,
    then the score is quiet large.
    """
    if target == "CUDA":
        time_ms = perf_bang.benchmark(file_name)
    elif target == "BANG":
        time_ms = perf_bang.benchmark(file_name)
    elif target == "DL Boost":
        time_ms = perf_dlboost.benchmark(file_name)
    if time_ms is None:
        return 0.0
    return GFLOPS / (time_ms / 1e3)


# 使用一个辅助函数选择对应的 Action
def get_action_from_space(action_id):
    return ActionSpace[action_id]  # 通过索引获取相应的 Action


# 使用 vmap 或 lax.map 实现映射
def dynamic_action_selection(cur_action_ids):
    return jx.vmap(get_action_from_space)(cur_action_ids)


class FalconGo:
    def __init__(
        self,
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=A_Length,
        optimizer_len=11,
        goal_reward=False,
        timeout=None,
    ):
        self.timeout = timeout
        self.file_name = file_name
        self.op_name = op_name
        self.source_platform = source_platform
        self.target_platform = target_platform
        self.action_len = action_len
        self.optimizer_len = optimizer_len
        self.best_reward = 0.01
        self.best_optimizer_ids = None

    def pick_best_annotation(self, actions):
        with tempfile.TemporaryDirectory() as work_dir:
            spaces = ms.space_generator.PostOrderApply(sch_rules=actions)
            database = ms.tir_integration.tune_tir(
                mod=self.mod,
                target=self.tvm_tgt,
                work_dir=work_dir,
                max_trials_global=32,
                num_trials_per_iter=16,
                space=spaces,
            )
        sch = ms.tir_integration.compile_tir(database, self.mod, self.tvm_tgt)
        if sch is None:
            return None
        else:
            return sch.mod

    def perform_action(self, actions):
        """Generates a design space for a given `action`. It calls `generate_design_space()`
        with specific parameters to apply the given scheduling rule (`action`) to the module.
        The function returns a new `ProgramState` object, which represents the new program
        state after applying the action."""
        with open(self.file_name, "r") as f:
            code = f.read()
            f.close()
        for action in actions:
            code = action(code)
        with open(self.file_name, "w", encoding="utf-8") as f:
            f.write(code)
        score = objective(self.file_name, self.target)
        return code, score

    @jit
    def step(self, action_id, env_state):
        # env_state 我需要得到如下信息。
        # -- optimize_path: 父节点所有位置。个数为访问深度
        # -- optimize_grid: 访问矩阵。矩阵是 action length * optimizer length
        # -- depth: 访问深度
        optimize_grid, trajectory, depth = env_state
        trajectory = trajectory.at[depth].set(action_id)
        cur_action_ids = lax.dynamic_slice(
            trajectory, (0,), (depth.val[0] + 1,)
        )
        cur_actions = [
            ActionSpace[_i]
            for _i in jx.device_get(cur_action_ids.val[0]).tolist()
        ]

        # try:
        _tvm_space, reward = self.perform_action(cur_actions)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_optimizer_ids = cur_action_ids.val[0].tolist()

        print(
            f""" Action: {cur_action_ids.val[0].tolist()} Reward: {reward} Best Reward: {self.best_reward} Best Reward IDs: {self.best_optimizer_ids}"""
        )

        # except:
        #     reward = 0.0

        optimize_grid.at[depth, action_id].set(True)
        # Treminated if we reach the goal
        terminal = depth > self.optimizer_len

        next_env_state = optimize_grid, trajectory, depth + 1
        return (
            next_env_state,
            self.get_observation(next_env_state),
            reward,
            terminal,
            {},
        )

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        optimize_grid = jnp.zeros(
            [self.action_len, self.optimizer_len], dtype=bool
        )
        trajectory = jnp.zeros(self.optimizer_len, dtype=int)

        depth = 0
        env_state = optimize_grid, trajectory, depth
        return env_state

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        optimize_grid, trajectory, depth = env_state

        return optimize_grid

    def num_actions(self):
        return self.action_len


def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def build_env(file_name, source_platform="BANG", target_platform="DL Boost"):
    action_len = len(ActionSpace)
    base_name = os.path.basename(file_name)
    op_name = base_name.split("_")[0]
    optimizer_len = 8
    tvm_env = FalconGo(
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=action_len,
        optimizer_len=optimizer_len,
    )
    return tvm_env


def _test():
    source = "BANG"
    target = "DL Boost"
    action_len = len(ActionSpace)
    optimizer_len = 8
    tvm_env = FalconGo(
        name,
        source,
        target,
        action_len=action_len,
        optimizer_len=optimizer_len,
    )
    optimize_path, optimize_grid, trajectory, depth = tvm_env.reset(None)
    optimize_path = [
        0,
    ]
    env_state = optimize_path, optimize_grid, trajectory, depth
    tvm_env.step(1, env_state)


if __name__ == "__main__":
    _test()
