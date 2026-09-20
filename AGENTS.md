# Probe 1 项目执行指令

本文件面向在现有 RoboTwin 项目中工作的编码 agent。任务是实现双臂 VLA 协调失败诊断实验，而不是从零编写机器人学习框架。

#### 前提：
项目在https://github.com/jia7788-c/RoboTwin_probes.git , branch是probe1

模型与产物（都在大盘 /home/data_ssd1/zyj/）
SFT 模型：/home/data_ssd1/zyj/openpi_model (使用hf镜像下载)
输出：/home/data_ssd1/zyj/RoboTwin_probes


## 1 目标与范围

研究对象为 SFT 后的 π0.5，在 RoboTwin 2.0 clean 环境中检验：

- H1：强耦合任务的协调失败风险高于弱耦合任务。
- H2：Joint π0.5 仍存在具有实际意义的残余协调失败。
- H3：在测试的数据规模和训练预算内，增加同分布演示后，协调失败的边际改善可能变小。

这些是假设，不是预设结论。不得为得到正结果修改任务集合、阈值或排除规则。数据平台期不能单独证明架构缺陷或显式结构的必要性。

当前阶段只做 Probe 1，不启动 RL，不实现新 critic，不全面复现 Probe 2/3/4。只为后续探针保留状态和接口。

## 2 代码归属与复用原则

以现有 RoboTwin fork 为运行入口，优先新增 `experiments/probe1/`。不要求先创建独立研究框架。

| 工作 | 归属 |
| --- | --- |
| 任务、物理环境、成功判定、专家数据 | 复用 RoboTwin |
| 策略接入 | 优先复用当前 checkout 的 adapter 或 XPolicyLab；先检查实际版本 |
| π0.5 SFT、推理、输入和模型内部改动 | OpenPI 工作副本或 fork |
| 仿真状态记录 | RoboTwin 侧最小 recorder/hook |
| 事件检测、失败分类、统计 | `experiments/probe1/`，尽量支持离线运行 |
| 后续 RL | 留给 RLinf，不作为本阶段依赖 |

不得根据旧教程臆造 `policy/openpi/` 等路径。优先 wrapper/hook；必须修改上游时保持补丁小且记录理由。不要静默更新子模块或更换框架版本。

## 3 开始编码前的检查

先做只读检查，输出简短环境审计，再决定实现路径：

1. 阅读现有项目指令、README、环境锁文件；检查 git 状态并保护用户改动。
2. 确认 RoboTwin、策略接入层、OpenPI 的路径、commit 和子模块版本。
3. 定位真实评测入口、任务注册、成功判定函数、策略 adapter 和配置读取路径。
4. 确认 Python 环境、GPU 型号与显存、用户可用设备、模型权重与数据是否存在。
5. 核实 checkpoint 是基础预训练还是目标任务 SFT；不能将基础模型低成功率直接解释为协调瓶颈。
6. 检查 head/wrist 图像顺序、RGB/BGR、尺寸、state 维度、左右臂顺序、动作单位、绝对/增量控制、夹爪方向和归一化。
7. 分别记录预测 horizon `H`、每次实际执行长度 `K`、控制周期和物理步长；不得写死 H=50 或 64。
8. 确认 contact、object pose、任务变量与状态恢复接口是否可用。缺字段应明确报告，不伪造默认值。

没有服务器或真机访问权限时只实现可验证的代码和离线测试，明确哪些仿真测试未执行。不得把 mock 测试写成真实实验结果。

## 4 第一轮任务与授权边界

默认先完成 M0：未修改的 Joint π0.5 + Handover Block + clean 单任务评测与最小日志。

- 从本地 registry 获取准确 task ID 和 embodiment ID；目标 embodiment 为 ALOHA-AgileX，不能凭名称推断资产兼容。
- 优先复用已验证的目标 checkpoint；没有时报告 SFT 需求，不擅自换模型。
- 先验证配置和一个 episode，再进行用户授权范围内的小规模 smoke test。
- smoke test 通过后交付记录，不自动启动 50-task 评测、长时间 SFT 或多卡实验。
- 开始昂贵作业前确认 GPU、任务数、训练 seed、episode 数、时长预算及输出路径。既有明确授权无需重复询问。
- 不终止他人进程，不占满所有 GPU，不重装已有环境，不覆盖 checkpoint 和实验结果。

## 5 建议新增模块

以下是目标结构，不代表文件已经存在。按照实际代码风格调整，避免重复实现上游能力。

| 相对 `experiments/probe1/` 的路径 | 职责 |
| --- | --- |
| `README.md` | 经过验证的安装、运行、分析命令与限制 |
| `configs/` | 任务分组、耦合评分、阈值、seed、实验预算 |
| `rollout/run_eval.py` | 复用上游评测循环的薄入口 |
| `rollout/recorder.py` | 状态、动作、接触、视频索引记录 |
| `analysis/event_detector.py` | 产生事件和证据，不直接宣称因果 |
| `analysis/failure_analyzer.py` | 任务规则下的 episode 主因分类 |
| `analysis/statistics.py` | 指标、置信区间、任务级汇总 |
| `analysis/plot_results.py` | 仅消费真实结果生成图表 |
| `tests/` | 单元测试、schema 测试及可选集成测试 |

大体积数据、视频、权重、快照放可配置的外部 output root，不提交 git。新增脚本应提供参数校验和 `--help`；路径不硬编码为开发者机器路径。

## 6 阶段与验收

| 阶段 | 工作 | 验收条件 |
| --- | --- | --- |
| M0 | 环境检查和 Joint 单任务 | 真实 rollout 完成，动作映射与成功标志已核查，版本与日志可追溯 |
| M1 | Handover 事件检测 | grasp/release/drop 事件可对齐视频，未观测事件正确处理 |
| M2 | 2 strong + 1 weak | 任务规则可复用；完成人工审计；无明显分类泄漏 |
| M3 | 12-task B0/B1 | 冻结任务与 seed；B1 设计经用户确认；正式评测与统计完成 |
| M4 | 50-task B0 筛查 | 报告所有任务及异常，不删掉低成功率任务 |
| M5 | 4–6 strong 数据缩放 | 嵌套数据、训练预算、重复 seed、结果曲线完整 |

任务选择应在查看正式测试结果前冻结。工程试点与正式测试使用分离样本；不得因 pilot 失败多而专门挑选任务。

## 7 模型与实验矩阵

### 基线

- B0：标准 Joint π0.5，优先保持模型内部不变。
- B1：移除另一臂 proprioception 的匹配消融，需要重新训练对应条件。仅推理时置零属于另一种分布偏移探针，不能替代 B1。
- B2：factorized heads，仅后续明确授权时实现。

B1 编码前绘制或列出真实信息通路：state/token 编码、共享视觉、动作 token、denoising、共享 hidden state。共享视觉仍包含另一臂信息，因此 B1 不得称为“完全无协调”或“完全独立”。分别遮蔽后两次推理再拼接动作会改变计算量和动作一致性，不能默认为公平实现。增加 missingness mask、拆 head 或增加参数时，说明设计变化并对齐对照。两套独立策略仅作经用户确认的备选，不称理论下界。

### 计划预算

下表为设计预算，不是自动运行授权；episode 数以每 task、每训练 seed 计。

| 实验 | 任务范围 | 条件 | 计划 episode |
| --- | --- | --- | --- |
| E0 | Handover Block | B0，校准用 | 约 200，先从小规模开始 |
| E1 | 50 tasks | B0 | 40；预先决定是否用 30 或 50 |
| E2 | 12 tasks | B0 | 100 |
| E3 | 相同 12 tasks | B1，与 E2 配对 | 100 |
| E4 | 4–6 strong | B0，25%/50%/100% demos | 每级 100 |
| E5 | 4–6 strong，可选 | B2 | 100 |

正式实验目标为 3 个独立训练 seed；单 seed 结果标为探索性。评测 seed 与训练 seed 分开记录。按 E4 五任务估算，单训练 seed 约 5,900 episodes，不含 E0/E5；重复使用满足完全相同协议的 B0 结果时去重并记录来源。

在 SFT 开始前明确单任务模型还是多任务模型，不能将两种口径混合。统一初始化、数据处理、增强、优化器、学习率、训练步数、batch、可训练模块、视角和评测协议。新 head 的初始化应单独记录。

## 8 任务耦合与数据管理

对 50 tasks 标注 C1 shared-object、C2 temporal dependency、C3 spatial dependency、C4 simultaneous activity，各为 0/1；`Ctask=C1+C2+C3+C4`。暂按 0–1 weak、2 medium、3–4 strong 分层。

- 该分数是操作性启发式，不是已验证的因果度量；同时运动不等于必须协调。
- 两名人工标注者盲于模型结果评分，保留原始分数、证据和仲裁记录。agent 不得伪造人工标注。
- 12-task 默认候选方案为 4 strong + 4 medium + 4 weak，或经确认采用 6+6；覆盖不同机制，按任务定义和专家轨迹选择。
- 计算双臂同时活动比例、共享接触比例、任务相关事件间隔作为补充；不把任意加权总分当作 ground truth。
- train/validation/test 按 episode 和场景 seed 隔离；不将同一轨迹的不同帧分进不同数据集。
- E4 使用固定绝对数量及嵌套 episode ID：25% ⊂ 50% ⊂ 100%，每层保存 manifest/hash。
- 明确主分析采用固定更新步数；记录有效 epoch、loss 和收敛状态。必要时单独报告收敛匹配补充，不能把未收敛误判为数据平台。
- 归一化规则预先确定。若使用全量训练集统计，披露小数据条件利用了全量统计；若各子集独立计算，记录其变化并考虑敏感性分析。不得使用测试数据计算统计。

## 9 日志合同

每次 run 保存机器可读 manifest：repo commits、配置 hash、checkpoint 标识/hash、数据 ID/hash、baseline、训练 seed、eval seed、依赖版本、动作和状态字段映射、H/K/频率、输出目录。

每条 episode 至少记录：

- `task_name`, `episode_id`, `run_id`, `eval_seed`, `success`, `termination_reason`。
- simulator step、控制 step、物理时间戳和视频帧对应关系。
- 左右 joint position/velocity、EE pose、gripper opening。
- 目标及相关物体 pose/velocity、接触对象/link ID、可用的接触力。
- 策略原始预测 chunk、反归一化 action、实际发送给控制器的 action。
- `chunk_id`, `chunk_start_step`, `action_index_in_chunk`, `executed_length`。
- policy 输入图像/指令的可追溯索引，task phase 及其判定来源。
- 分析标签、事件时间、规则版本、置信等级及 evidence。

坐标系、四元数顺序、单位、字段 shape 必须写入 schema。缺失字段使用 null/availability 标记，不能用 0 假装观测值。

仿真异常与策略失败分开：异常记录为 infrastructure error，按预注册规则重试同 seed；超时属于有效策略失败，不能随意排除。分析中报告尝试数、有效 episode 数和排除原因。

### 状态恢复

普通 qpos/object pose 日志不等于完整快照。检查物理状态、控制器缓存、任务变量、随机数状态、策略动作队列和推理随机性。仅当“保存—恢复—相同动作重放”的数值误差与终局结果通过测试时，标记 `replay_verified=true`。否则记录 `false` 和限制，不能宣称已具备 Probe 4 的反事实回滚能力。

## 10 失败分类和事件判定

成功 episode 的 `primary_failure=null`。失败 episode 恰有一个主标签：

| 标签 | 证据要求 |
| --- | --- |
| `perception_grounding` | 明确选错对象/区域等证据；动作错误不能直接证明内部感知错误 |
| `left_execution` | 左臂局部执行违规，无法由更早的协作前置条件违规解释 |
| `right_execution` | 右臂同上 |
| `temporal_coordination` | 任务定义的跨臂顺序/时序约束被违反 |
| `spatial_coordination` | 任务相关相对位姿/共享物体约束或非期望碰撞证据 |
| `other_uncertain` | 多原因难区分、缺证据或无法规则化 |

优先记录最早有证据的失败事件；日志判定是诊断性归因，不等于干预证明的真实因果。多个候选同时成立时保留候选证据，无法仲裁则进入 uncertain。

`object_dropped`、`collision`、`timeout` 等是可多选的 outcome；不能与互斥主因相加。成功轨迹中可记录可恢复违规，但不能计入终局 CFR。

事件规则：

- Stable grasp：综合持续接触与物体相对夹爪的稳定性；移动物体合法向下运动时，不能仅因下降否定抓持。
- 持续窗口先用秒定义，再按实际采样周期换算；记录事件起点与确认时刻，避免 k 帧确认延迟混淆 timing。
- Release：结合夹爪开度和接触消失，抑制抖动。
- Handover：`delta_t = t_release_donor - t_stable_grasp_receiver`；`delta_t < -delta` 为候选提前释放。接收手从未抓住时 delta_t 为 null，按任务规则处理，不伪造时间。
- Spatial：使用阶段相关的相对目标；平移以米、旋转以弧度分别阈值化，或定义有明确尺度权重的距离，不能直接混合单位。
- Collision：核查 link、力、持续时间与阶段，区分任务允许接触。
- Drop：仅标记非期望失去支撑；预期放置/释放不能判为掉落失败。

阈值从独立 pilot 标定，记录来源并在正式测试前冻结。未标定项配置为 null，正式分析必须 fail fast，不能静默使用示例数值。

## 11 人工审计

分开 detector 开发集和冻结审计集。抽取至少 10% 失败，并尽量覆盖每个 task、主要类别及 baseline；稀少类别补充抽样时记录采样权重。抽查成功轨迹，检验误报。

两名人工标注者独立查看视频与状态，隐藏模型身份和自动标签，仲裁后形成参考标签。报告 κ、Accuracy、Macro-F1、各类 precision/recall、样本数和混淆矩阵。

工程目标为 κ≥0.80、Macro-F1≥0.85；Other>15% 时暂停扩展并检查。它们是项目质量门槛，不是通用统计定律。反复根据审计集调参后，该集只能算开发数据，需新的留出审计集验证。

## 12 指标与统计约束

对每个 task × baseline × training seed 输出原始计数以及：

```text
SR = N_success / N_valid
CFR_all = (N_temporal + N_spatial) / N_valid
CFR_fail = (N_temporal + N_spatial) / N_failure
Delta_CFR = CFR_all_strong - CFR_all_weak
Delta_SR = SR_joint - SR_blind
```

分母为零返回 null，并报告分母。主汇总采用 task-macro average；episode-micro average 作为补充并明确命名。CFR_all 和 CFR_fail 同时报，不能只凭剩余失败比例上升宣称绝对风险不降。

- H1 主要报告 Delta_CFR 和按 task 聚类的 95% bootstrap CI；模型比较保留相同 seed 的配对关系。
- 训练 seed 与 episode 重复不等价。汇总考虑任务、训练 seed 和 episode 层级，不将所有 rollout 当独立样本。
- 可选 mixed-effects logistic 回归检查 coupling 与 baseline interaction，检查收敛、完全分离和任务数不足问题。
- coupling 与任务难度、长度、对象数可能混杂；记录这些描述量，相关性不写成因果证明。
- 不以两组 CI 是否重叠代替差值检验。
- H2 的“实际重要”阈值在正式测试前由用户确认，不凭 `CFR>0` 声称主要瓶颈。
- H3 预先确定最小有意义改善 epsilon，估计 50%→100% 的 CFR_all 改善及 CI。区间过宽或仅 p>0.05 时结论为不确定，不是平台。3 个数据水平不足以外推无限数据结论。
- 报告 effect size/CI、探索性与确认性检验；任务级多重比较按预注册方案控制 FDR。

## 13 测试要求

离线单元测试至少覆盖：

1. 成功无主因，失败主因互斥；多个 outcome 不重复计数。
2. 接触抖动、提前释放、合法放置、接收方从未抓住、证据不足。
3. 左右臂交换后规则对称；时间窗口边界和采样频率换算正确。
4. 缺失 contact/pose 时进入明确缺失处理，不误判成功或零误差。
5. N_failure=0、全部失败、空结果、重复 episode ID。
6. JSONL/schema 校验、manifest/hash 保存和断点续跑不覆盖旧记录。
7. bootstrap seed 固定时结果可复现；任务配对不被打乱。

集成测试至少覆盖真实单 episode 日志可读、动作映射正确、标签对齐视频。状态恢复单独设测试；不可用时明确 skip 原因。使用 synthetic fixtures 时清楚标注，不混入正式分析。

## 14 交付与决策

每轮只交付已完成阶段，说明：改了哪些文件、如何运行、测试是否真实通过、未验证项、下一阶段条件。运行命令必须来自实际入口验证；未验证命令显式标注，不冒充可直接运行。

最终 Probe 1 交付包括任务分组与依据、B0/B1 结果、人工审计、数据缩放、可复现统计、任务级表和核心图：coupling vs SR、failure taxonomy、Joint vs blind、data scaling、handover timing。

决策允许 Strong Go、Weak Go、No-Go 和 Evidence insufficient。若 coordination 存在但跨臂利用充分，需结合后续 Probe 2 再判断是否转向 credit/recovery，不能由 Probe 1 单独宣称模型已充分建模跨臂信息。

禁止事项：

- 不重写已有框架，不擅自引入 RL，不未经确认训练 B1/B2。
- 不把 50 个任务都认定为强协同，不根据正式结果挑任务。
- 不伪造人工审计、运行结果、路径、API、checkpoint 或参考数值。
- 不以日志关联替代严格因果结论，不把数据平台直接解释为架构必要性。
- 不把本指令文件本身视为长时间 GPU 作业、删除数据或修改外部仓库的授权。

第一轮应交付：环境审计、最小接入方案、真实单任务 smoke test（条件允许时）、最小日志与对应测试。其余阶段在此基础上逐步推进。
