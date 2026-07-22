# train 目录结构

`train/` 是 Python 训练上层。它负责把 Go2/controller 交互包装成 gym 风格环境，调度 collector 和 learner，并通过 `rl/` 下的算法 backend 完成更新。算法底层不放在这里。

## 入口层

- `__main__.py`：`python -m train` 入口，只转发到 `main.py`。
- `main.py`：CLI 参数解析，读取单一 `config/go2.yaml`，选择 `in_process` 或 `play`。
- `config.py`：解析 `config/go2.yaml`，生成 robot/train/agent 配置对象。

## 训练编排

- `learner.py`：训练模式编排。负责创建 env、agent、replay，运行 `in_process`、`play`。
- `loop.py`：单进程在线训练循环。保留在独立文件，避免 `learner.py` 过大。
- `checkpoint.py`：完整训练 snapshot，包括 agent、replay、step、metadata。
- `warmup.py`：JAX/PyTorch backend warmup。

## Go2 环境适配

- `env.py`：Go2 gym 风格环境、controller IPC 调用、reset/standup/recovery、action filter。
- `gym_env.py`：gym/gymnasium API、obs/action space、dtype、flatten/rescale 等通用包装。
- `dds.py`：DDS low state / sport state reader。
- `ipc.py`：controller Unix socket client。
- `obs.py`：四元数、姿态稳定性和 belly-up 等底层传感器 helper。
- `types.py`：Go2 状态数据结构。

## MDP Core

- `mdp/spec.py`：observation/action/reward/termination 等 MDP spec 的版本化元数据。
- `mdp/observation.py`：统一 observation 构造。
- `mdp/action.py`：统一 normalized action 到关节目标的映射。
- `mdp/reward.py`：统一 reward rate、dt scaling 和 reward info。
- `mdp/costs.py`：可部署 safety cost component。
- `mdp/termination.py`：统一 terminated/truncated 语义和去抖终止条件。
- `mdp/transition.py`：把 env step 结果转换成 `common.transition.Transition`。

MDP Core 必须保持 DroQ 与 FlashSAC 共用；算法对比实验只能改变 backend、exploration、replay/update 机制、网络结构和算法超参数。

## Collector

- `collector/legacy_env.py`：当前 in-process 路径使用的 legacy env builder。

Collector 只产出观测、动作和 transition，不直接依赖 `rl.droq.data` 或 `rl.flashsac.buffers`。具体 replay/buffer 写入由 `learner.py` 或算法 backend 决定。

## 日志和指标

- `logging.py`：可选 wandb logger。
- `profiling.py`：step/update/loop timing。
- `rolling_metrics.py`：滚动训练摘要。

## 辅助入口

- `recovery.py`：手动触发 recovery/standup 的维护入口，不参与默认训练链路。

## 新增文件规则

优先扩展现有文件：

- controller、DDS、IPC、reset、Go2 runtime 逻辑放 `env.py`、`dds.py`、`ipc.py`。
- observation/action/reward/cost/termination/transition contract 放 `mdp/`。
- 训练循环、update 调度、checkpoint、日志指标放 `learner.py`、`loop.py`、`checkpoint.py` 或指标文件。
- 算法实现、replay buffer、loss、exploration 不放 `train/`，应留在 `rl/droq` 或 `rl/flashsac`。
- 临时分析文档放 `.agents/`，不要放进 `train/`。
