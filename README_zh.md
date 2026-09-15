# unilab-rl

[![PyPI](https://img.shields.io/pypi/v/unilab-rl)](https://pypi.org/project/unilab-rl/)
[![CI](https://github.com/unilabsim/unilab_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/unilabsim/unilab_rl/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[English](README.md) | 简体中文

从 [UniLab](https://github.com/unilabsim/UniLab) 拆出的强化学习算法与异步
runtime，打包为独立的、与仿真器解耦的库。

- distribution 名：`unilab-rl`
- import namespace：`uni_rl`
- 仓库：[unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)

## 与 UniLab 的关系

`uni_rl` 是 UniLab 项目的 RL 算法与异步 runtime 层，独立拆分成本包。
[UniLab](https://github.com/unilabsim/UniLab) 仍是消费方：它持有物理仿真
backend、任务套件与训练入口，并通过 `uni_rl.env_contract.EnvFactory` 把
env 注入 `uni_rl`。`uni_rl` 永远不 import `unilab` / `unisim`，也不自行
构造 env——任何满足 contract 的向量化环境（包括 UniLab 之外的仿真器）都
可以驱动本包中的算法。

如果你用 UniLab 训练，`uni_rl` 会作为依赖自动带入；只有当你想把这套算法
与异步 runtime 复用到自己的环境栈时，才需要直接安装 `unilab-rl`。

> 命名说明：最初想要的 distribution 名 `uni-rl` 在 PyPI 上不可注册
> （与既有 `unirl` 项目 ultranormalized 冲突），因此以 `unilab-rl`
> 发布；import namespace 保持 `uni_rl` 不变。

## 内容

- **On-policy**：基于 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
  的 PPO（`FinalObservationAwarePPO`、`RslRlVecEnvWrapper`）
- **异步 PPO（APPO)**：原生 collector/learner 多进程实现
- **Off-policy**：FastSAC、FastTD3、FlashSAC，配 double-buffer 异步 runner
- **Runtime 基础设施**：共享内存 rollout/replay buffer、replay pipeline、
  DP 梯度同步、显存预算、tensorboard/wandb 训练 logger 与 trace recorder

## 目录结构

- `uni_rl.algos.*` — 算法层：on-policy(`rsl_rl` PPO 封装）、异步
  on-policy(`appo`)、off-policy learner(`fast_sac`、`fast_td3`、
  `flash_sac`）与共享算法辅助（`common`)
- `uni_rl.ipc` — runtime 基础设施：异步 runner、共享内存 rollout/replay
  buffer、replay pipeline、DP 梯度同步、显存预算
- `uni_rl.offpolicy` — 通用 off-policy double-buffer runner 脚手架
- `uni_rl.logging` — tensorboard/wandb 训练 logger、trace recorder
- `uni_rl.utils` — device、seed、nan-guard、观测辅助
- `uni_rl.env_contract` — 注入式 env factory/protocol contract

## 安装

```bash
pip install unilab-rl
# 或使用 uv：
uv add unilab-rl
```

要求 Python 3.10–3.13,PyTorch ≥ 2.7。

## 使用

`uni_rl` 不构造 env。请把可 pickle 的 env 工厂
(`EnvFactory = Callable[[int, Mapping | None], EnvProtocol]`）注入所选
算法的 runner:

```python
from collections.abc import Mapping

from uni_rl.env_contract import EnvProtocol


def make_env(num_envs: int, cfg: Mapping | None) -> EnvProtocol:
    """顶层工厂函数（可被 pickle 引用；禁止闭包/lambda）。"""
    ...
```

env contract 是一个最小化的、基于 numpy 的自动 reset 向量化环境协议：按
观测组键控的 dict 观测（`obs_groups_spec`)、带 final-observation 语义的
`step()`，以及返回 `(obs, info)` 的 `reset()`。完整 contract 见
[`src/uni_rl/env_contract.py`](src/uni_rl/env_contract.py) 的模块
docstring；如何不 fork 本仓库、通过 `runtime_resolver` 接入自定义算法，
见 [`AGENTS.md`](AGENTS.md) 的「新算法扩展方式」一节。

### PPO curriculum checkpoint 状态

`RslRlPPORuntime.runner_cls` 是可选的：训练入口可藉此在 wrapper 之外选择
自定义 runner。取 `None` 时保持入口既有的标准 runner；旧的仅提供 wrapper
的 resolver 继续可用。

对于必须恢复 curriculum 进度的训练，请选择
`uni_rl.algos.rsl_rl_training_state.TrainingStateOnPolicyRunner`。其 wrapper
必须显式实现 `uni_rl.training_state.TrainingStateProvider` 协议，否则调用方
必须向 runner 传入 `training_state_provider=`:

```python
def export_training_state(self) -> Mapping[str, object]:
    return {"schema": "my-task-v1", "steps": self.steps, "difficulty": self.difficulty}


def import_training_state(self, state: Mapping[str, object]) -> None:
    # Validate the complete owner schema before changing any state.
    ...
```

runner 会在 `checkpoint["infos"]["uni_rl_training_state"]` 中存放一个
version-1 envelope;payload 为纯 JSON 数据，schema/version 归 provider
所有。数组必须显式转换为 list。runner 不会探测嵌套的 env，也不会序列化
owner 对象。算法、optimizer、iteration 与 logger 的既有行为仍由父 runner
负责。

`load()` 默认要求存在有效的 training state。只有显式的 actor-only
`load_cfg={"actor": True}` 才允许对 legacy checkpoint 使用
`restore_training_state=False`。envelope 错误会在算法加载之前被拒绝；
provider 的 import 错误会直接传播，且必须中止 resume。算法与 provider 的
加载不是事务性回滚。本 contract 覆盖训练进度，不包含物理或 RNG 快照。

该 runner 独立于任何具体任务或仿真器。下游 provider 必须把自身的计数器与
自适应 curriculum 状态一起恢复；如果下一步会重新计算某个派生计数器，只
恢复它本身是不够的。

## 设计契约

`uni_rl` **不**依赖任何仿真器或环境库。算法行为归属 `uni_rl.algos.*`
下的各算法模块；runtime 基础设施（`ipc`、`logging`、`offpolicy`、
`utils`、`env_contract`）位于顶层，且不依赖算法层。env 侧参考集成见
UniLab 的训练入口。

## 开发

```bash
make sync      # 安装依赖(uv)
make test      # pytest
make format    # ruff check --fix + ruff format
uv run mypy src/uni_rl && uv run pyright   # 类型 gate
```

## 引用

如果在研究中使用了 `unilab-rl`，请引用 UniLab 论文：

```bibtex
@article{jia2026unilab,
  title   = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author  = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal = {arXiv preprint arXiv:2605.30313},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.30313}
}
```

## 许可证

Apache-2.0，与 UniLab 一致。
