# 遥感任务挑战、系统缺口与下一轮实施计划

更新：2026-09-23。审计代码基线：`acb5c4f`，其 A800 实现父提交为 `0c81784`。
范围：公开一手资料核验、代码审计、实验设计；本次不启动 GPU 实验，也不修改运行逻辑。

## 1. 判断与证据边界

下一轮应优先建立**能严格评分的遥感决策任务**：在空间覆盖、时间、云遮挡、分辨率和预算限制下，Agent 必须选择信息、执行分析、核对证据，再回答或拒答。增加地图图层或数据集数量只能扩大输入范围，不能单独证明推理和规划能力。

已有工程基础可以继续复用：任务冻结、scoped gateway、栅格对齐、固定公式、时序筛选、证据记忆、隐藏真值评价、幂等、执行重放都已存在。缺口主要是验收检查的严格性、真实语义评价的覆盖、需要自主决策的任务，以及跨地区和时间的实验。

### 上轮结果的准确解释

- 2026-09-22 的报告记录 11 个**数据集接入条目**、每条 2 个样本，最终 22/22 交互通过，74 次模型生成的 action/function call。DOTA、XLRS 的多个接入条目不能直接视为独立数据来源或独立任务能力。
- 22/22 是修复并重试后选定报告的结果，不是无干预首轮成功率，也不是语义准确率。两例只能做接入检查，不能估计 dataset-wide 表现。
- 全量 unittest 输出为 `Ran 382 tests ... OK (skipped=34)`，应写作 **382 项总数、348 通过、34 跳过、0 失败**；此前“382/382 通过，另有 34 跳过”计数错误。renderer 为 4/4。
- 本次发现汇总器任务/资产匹配有空检查（见 G01）；既有 JSON 中的 `real_model_acceptance=true` 保留为历史产物，不能视为通过修正后严格验收的证明。需补检查后用原始证据重新审计，不能预先承诺仍为 22/22。
- SQLite 完整性、artifact 元数据绑定、网络隔离、答案正确、模型自主选择、故障恢复是不同验收维度，结果应分列。

## 2. 遥感任务真正困难在哪里

| 挑战 | 具体失败形式与一手依据 | Harness 应实现或验证的要求 |
|---|---|---|
| 大范围与小目标并存 | 整图缩放丢失小目标；旋转目标需 OBB 而非普通矩形。DOTA 定义 OBB、difficult 标记，并允许 GSD/日期缺失 [S1]；GEOBench-VLM 提供计数、定位和时序等任务分类 [S9] | overview→候选区域→高分辨率窗口；记录有效 GSD、缩放与坐标变换；跨窗口去重；缺 GSD 时不得输出伪造的平方米/米 |
| 地理坐标与波段物理意义 | 波段、CRS、分辨率不同，空间对齐后仍可能有单位、scale/offset、NoData 和重采样错误。TorchGeo、rslearn 明确处理这些异构输入 [S2,S3] | 保留现有 grid/scale/mask 检查；增加误配波段、轴顺序、跨格网和缺元数据负例；分类掩码与连续反射率分别重采样 |
| 时间、可观测性与变化混淆 | 洪水判断需要灾前结构和灾后状态；“两图不同”可能只是日期、视角、云或季节不同。SpaceNet 8 明确联合灾前/灾后影像与道路、建筑标签 [S4] | task 固定事件时间窗、候选可用性和空间重叠；把“没观测到”“无变化”“证据不足”分开；限制未来影像进入当前决策 |
| 云质量策略不是真值 | CloudSEN12 同时涉及云和云影 [S5]；本项目 8-window 审计已找到 SCL 阈值接受但独立标签判为质量不足的反例 [I3] | 提供质量与有效覆盖报告，人工标签只给 evaluator；报告误接受、误拒绝和拒答质量，不能用同一 SCL 输出既筛选又证明正确 |
| 多模态与不确定性传播 | 光学、SAR、DEM、矢量有不同观测量；STAC 能描述多种资产，不能保证这些资产已经正确配准或适用于任务 [S6] | 首先将传感器、处理级别、单位、时间和配准条件纳入契约；SAR/DEM 工具按实际问题接入，不能把所有输入当 RGB 图 |
| 标注与问题定义的限制 | DOTA 标记 difficult；不同标注版本重标了相同影像 [S1]。mask、类别、面积和自然语言答案不是同一个目标 | 明确 answer schema、真值版本、有效像素分母和不可评分状态；统计未知/歧义样本，不用文本流畅度代替结果正确 |
| 跨地区、跨时间泛化 | WILDS 的 FMoW 按年份与地理区域评估，PovertyMap 按国家及城乡分组 [S7] | 按原始 scene/AOI/event 分组切分，隔离相邻 tile、重叠窗口、同事件日期及重复源影像；报告宏平均和最差组 |
| 多步决策和证据复用 | rslearn 已能形成时空窗口与多源层，GEO-Bench-2 聚焦多模态/多时序模型评估 [S3,S8]；接入这些数据仍不会自动形成交互任务 | 用具有分支的信息缺口构造 episode；比较固定流程、无记忆 Agent、有记忆 Agent，衡量新增观测是否改善决策 |
| 数据与软件可复现 | DOTA 明示学术用途及商业限制；项目 EO-Gym 锁文件标记软件许可未解决 [S1,I4] | 每个输入保持来源、条款、内容 hash、处理链；公开代码与可再分发数据分别审核，避免将“可下载”视为“可发布” |

上表中的 Harness 要求是本项目设计建议，外部资料并未验证这些建议在本系统中有效。本轮是有针对性的资料调研，不是穷尽所有遥感任务的系统综述。

## 3. 现有系统的具体缺口

| ID / 优先级 | 当前实现及证据位置 | 缺口与下一步验收 |
|---|---|---|
| G01 / P0 验收绑定 | 已在 `2f0b92b` 修复：汇总器按权威 Pydantic 形态重建 manifest hash，并绑定 task/version、逐输入资产与内容 hash、episode、执行证据与模型 receipt；原始 22 例只读重审计通过 | 后续新增样本直接继承该严格验收；G01 从活动 TODO 移除 |
| G02 / P0 验收独立性 | `59612a0` 起逐 episode 校验 action/result/client_action_id 唯一且成功、每个 tool action 与唯一 completed tool_run 一一对应、tool request/action 和 expected_state_version 一致；`e028373` 起核验 final answer 引用无重复、全部来自权威 v2_evidence，每个 evidence 由成功 save action 创建且 source_ref 绑定 episode artifact；`50ca093` 起 Qwen transcript 保存 compact model request receipt（模型、参数规模、消息/tool 数、逐图像 decoded-payload SHA-256），验收要求 receipt 图像 hash 绑定 episode artifact hash。空/缺失/错误 outcome fail closed，重复 episode ID 有负例 | 原始模型响应与可信 adapter 变换的完整留存策略仍需后续运行包落地；旧 reports 无 receipt 时应 fail closed 而不能降级 |
| G03 / P0 科学评价覆盖 | 已在 `350e603` 汇总 semantic outcome：4 例有 evaluator、18 例明确 `unscored`，仅 1 例 task-correct；同时输出模型调用/token 成本 | 后续为 DOTA OBB、grounding、计数与 caption 定义独立 evaluator；不得把 `unscored` 计入正确 |
| G04 / P0 恢复、图像和成本证据 | 已在 `350e603` 增加 runner 接收 bytes SHA-256 校验、checkpoint 成本账本、attempt phase 和 resume 累计成本；终态 resume 不再覆盖原 token/调用统计 | 尚需真实/模拟“服务器已提交、客户端未收到响应”的中途恢复测试；本轮已覆盖终态 resume 与 checksum fail-closed |
| G05 / P1 自主信息获取 | 22 例大多是固定 crop→evidence→submit；任务、prompt 已规定动作与范围 | 构造必须选候选日期、空间窗口或替代输入的分支；错误区域、遮挡、空查询需触发有用的下一动作；与固定流程在相同权限/预算下比较 |
| G06 / P1 地理质量覆盖 | `raster_grid.py`、`raster_math.py`、`temporal.py` 已校验 reviewed grid、mask、scale/offset | 缺的是跨配置验收，不是重写对齐工具。补缺日期/GSD、云/NoData、格网偏移、类别插值污染和数值容差 cases；超出工具支持范围返回明确不可用 |
| G07 / P1 记忆收益与污染 | 2026-09-26 `runtime/evidence-memory-matrix-v5-20260926/matrix-run/matrix-manifest.json` 完成真实 Qwen 12-job 矩阵：correct-only 与邻区均检索唯一目标记录并正确提交；expired/date/sensor mismatch 均返回 0 后正确 abstain；conflict 返回 2 条后通过 `answer.abstain` 说明冲突并终止。报告含逐 episode 检索 ID、action、token 与时延；`5d3efea` 起 no-valid-memory abstention 使用 `expected_outcome=abstained` 并可得全分 | conflict 的 abstention 是保守正确行为但未加分，仍需显式冲突裁决或优先级语义；当前矩阵仍是单 AOI/单标签开发集，不能外推为统计收益 |
| G08 / P1 泛化与公平比较 | `config/qwen-dataset-matrix-v1.json` 是固定两例接入矩阵；runner 固定 `Qwen3.5-9B` 和参数 | 新建按 scene/event 分组的开发/验证/保留测试 manifest；记录模型 revision、视觉预处理、工具白名单和预算。保留每次失败与重试，分别报告 first-attempt 与 assisted completion |
| G09 / P1 可运行性与发布 | `config/eo-gym-source.json` pin 上游 revision 且许可 unresolved；本轮访问其中仓库地址得到 HTTP 404 | 404 只说明当前公开入口不可访问，不能推断原因。核验原取得源码的校验值、license 和可复现获取路径；保留自有 adapter，未解决许可前不再分发上游源码/数据 |
| G10 / P2 生产运行 | 已有 scoped token、mTLS、配额、物理用量审计和 runtime identity | 下一阶段才做恢复演练、分布式并发、外部证书生命周期和脱机审计锚定。已有 physical audit 和 runtime identity 从 TODO 删除，勿重复实现 |

G01–G04 是本次静态审计发现或待证明边界，不等于已经出现结果篡改、标签泄漏或错误图像。原始 22 条真实交互仍有价值；需要修复的是把它们提升为严格验收结论的检查过程。

## 4. 先做哪三类任务

| 任务包 | 复用输入与能力 | 隐藏评价及关键反例 | 主要指标 |
|---|---|---|---|
| 变化判断与可回答性 | 现有 WHU 两日期图像、crop、双源 evidence、change evaluator | 以已有人工编辑 masks 计算类别/方向/比例；加入缺时相、错误区域、受控遮挡。合成扰动只测健壮性，不能冒充新的真实传感器数据 | 类别 macro-F1、方向准确率、变化比例 MAE、false-confident rate、risk–coverage |
| 遮挡下的时序选择 | 已有 Sentinel-2 与 CloudSEN12 样本、quality policy、temporal select、NDVI | 明确只在独立标签覆盖的窗口上评价云质量；无独立覆盖的日期只测协议。反例：低 SCL 云量但无可用观测、无完整 AOI、日期错位 | 选中可用影像率、误接受率、AOI 有效覆盖、数值误差、搜索/工具成本 |
| 大图多尺度查找 | 已准入的 DOTA/grounding 影像，待补可核验 annotation 绑定 | 旋转框、边缘小目标、相邻窗口重复实例、空目标区域；原图级隔离 train/test | OBB mAP 或 grounding IoU（按原任务），小目标召回、重复率、像素/token 预算 |

先完成变化判断任务包，复用最多且隐藏真值路径已存在。SpaceNet 8 的洪水、道路、建筑组合适合作为后续外部扩展；本轮不下载或声称已接入。SAR/DEM、复杂道路拓扑也待具体评价目标确定后再开发。

## 5. 可执行里程碑与验收门槛

门槛为拟定的工程/实验准则，均非已获得结果。重要里程碑独立 commit，验收通过后可加 Git tag；实现不另开版本目录。

| 顺序 | 产物与修改点 | 完成条件 | 建议提交标题 |
|---|---|---|---|
| A / P0 严格验收 | 修正 result summarizer、receipt 与 fixture；read-only 审计原 22 份 report/SQLite/task/source manifest | 错 task/hash/样本/时相、失败 tool、空矩阵、损坏 snapshot 全部被拒绝；输出逐项原因；新报告单独存放 | `fix(acceptance): verify task and asset provenance` |
| B / P0 科学结果 | 汇总已有 evaluator 分数与 unscored；记录每次尝试、首轮成功、失败类型、累计 token/耗时；补中途故障恢复测试 | 交互成功且语义错误的反例不能进“任务成功”；终态 resume 不覆盖原运行统计；恢复不重复收费 | `feat(evaluation): report semantic outcomes and run costs` |
| C / P1 单实现布局 | 已在 `3e5bd2e` 完成：活跃实现迁至 `harness_api/app/core/`，`v2` 只保留转发 shim，协议路径/表/序列化名不变 | frozen HTTP fixtures 通过（V1 5/5，V2 4/4）；全量容器测试与迁移前基线同为 385 total / 5 fail / 3 error / 34 skip；payload guard 通过 | 已提交：`refactor(core): organize runtime by responsibility` |
| D / P1 决策任务包 | `71b953a` 完成 oracle answerability episodes；`994d006` 完成首个 Qwen 对照并暴露 runner 丢调用问题；`8185646` 修复为同轮多个 Qwen tool calls 按状态顺序全部执行；`b2cfe0b` 将 evidence 唯一性改为 `(episode_id,evidence_id)`；`1b5f3df` 冻结 40 例 answerability 开发包；本轮新增可恢复逐例 Qwen batch runner 与增量 manifest/分层摘要 | 低覆盖真实 episode 已证明正确 abstention；可回答历史失败保留：首轮 `unknown_evidence`，修复后首个 episode 因全局 evidence ID 约束失败；storage fix 后真实 Qwen run 已保存双证据并提交。开发包含 30 例 expected-submitted、10 例 expected-abstained、80 个 hash-pinned 公共输入和 80 个隐藏标签。batch runner 15 项 Qwen 相关测试 OK；下一步执行 40 例并按 scene/event 统计 | 下一个建议：`feat(qwen): add WHU batch runner` |
| E / P1 记忆实验 | 接入已有 memory，加入冲突与过期条件，统一 direct/fixed-pipeline/agent 的资源计量 | 报告共同准确率、错误率、成本及按 AOI/event 的配对置信区间；保留不提升或退化结果 | `feat(benchmarks): evaluate governed evidence memory` |

实施次序是 A→B→C→D→E；命名重构与科学行为变化分开提交。每阶段在 A800 验证、提交后再同步本地，不累计到一个无法定位问题的大提交。

## 6. 实验约束与研究问题卡

**问题：** 在相同数据权限和预算下，质量感知的信息选择与有来源的跨任务记忆，是否能提高不完整观测下的变化判断，并减少无依据回答？

**假设：** Agent 可通过补充观测提高可回答任务的准确率；对不可回答任务降低错误自信。记忆只有在时空范围、时效和来源满足条件时才有收益。

**当前证据：** 数据/工具/评价器与记忆生命周期已有实现；22 个模型交互记录证明基本连通；operator 驱动的 memory 验收证明受控检索路径。尚无匹配的自主模型消融结果。

**比较组：** 固定输入直接回答、固定分析流水线、无记忆 Agent、有治理记忆 Agent；另用错误/过期记忆作干扰条件。direct baseline 使用同样可用初始观测，Oracle/全信息结果只能作为上界，不能和受限输入混作公平对比。

**样本与统计：** 两例接入集保留作 smoke。第一轮建议按任务包准备 30–50 个可审查开发 episode，覆盖正常、遮挡、缺时相、错误候选和不可回答情况；这只是排障规模。正式测试按独立 AOI/event 数量和 pilot 方差确定样本量，预先固定 split、停止条件及重复运行规则。置信区间按 AOI/event 聚类，不能将像素或同一原图的 tile 当独立样本。

**主要指标：** semantic accuracy/F1、false-confident rate、risk–coverage、证据支持率、工具失败率、重复检索次数、输入像素/字节、总模型调用/token、端到端耗时。返回 confidence 只有经过校准评价才有可信解释；同时保留原始分数向量，aggregate reward 不作为唯一结论。

**支持条件：** held-out 集共同语义指标改善或在预定非劣界内明显降低成本，同时错误自信未上升；报告区间，不仅点估计。

**否证条件：** 增益只在调过 prompt 的两例中出现；去掉信息泄漏后消失；仅获得 treatment 专属评分奖励；更换地区或加入过期记忆后显著恶化；同预算固定流程表现相当或更好。

**最小下一步：** 完成 A，对 22 条存量证据重审；之后冻结 WHU 变化与可回答性开发包。此卡状态为“计划实验”，不声称方法有效或已有论文贡献。

## 7. 外部依据与核验记录

核验日期：2026-09-23。以下为本轮实际读取的官方数据/项目/规范正文；不是只读 abstract 的论文结论。动态页面以本次读取内容为准，正式实验还需固定仓库 revision、数据版本和许可。

| ID | 原始来源 / 证据类型 | 本文支持范围与限制 |
|---|---|---|
| S1 | [DOTA 官方数据定义](https://captain-whu.github.io/DOTA/dataset.html) / dataset | OBB、difficult、GSD/日期可缺失、标注版本和学术用途条款；不证明当前 Parquet 完整保留标签与元数据 |
| S2 | [TorchGeo 官方仓库](https://github.com/torchgeo/torchgeo) / software documentation | 多波段/CRS/分辨率、大影像采样、数据集交并；不能替代 Agent 执行审计 |
| S3 | [rslearn Core Concepts](https://github.com/allenai/rslearn/blob/master/docs/CoreConcepts.md) / software documentation | 时空 window、raster/vector layer、prepare→ingest→materialize；本项目尚未接入该库 |
| S4 | [SpaceNet 8 官方任务说明](https://spacenet.ai/sn8-challenge/) / dataset | 洪水道路建筑、灾前灾后和地理泛化场景；数据条款单独审核，不推断模型表现 |
| S5 | [CloudSEN12 官方仓库](https://github.com/cloudsen12/cloudsen12) / dataset documentation | 云/云影、L1C/L2A 与 temporal support；本轮 Nature 正文被页面加载拦截，HF 页获取失败，未将其记为已读全文 |
| S6 | [STAC specification](https://github.com/radiantearth/stac-spec) / standard | 时空资产元数据、Item/Collection 和扩展机制；不保证传感器物理一致性或科学分析有效性 |
| S7 | [WILDS datasets：FMoW、PovertyMap](https://wilds.stanford.edu/datasets/) / benchmark specification | 时间/地区/国家分布变化与最差组评价；本文未援引未经本轮核验的论文性能数值 |
| S8 | [GEO-Bench 官方说明](https://github.com/ServiceNow/geo-bench)、[GEO-Bench-2](https://the-ai-alliance.github.io/GEO-Bench-2/) / benchmark documentation | 当前官方推荐后继基准；后者为 fine-tuning-based 多模态多时序评估，不能当作 Qwen 零样本 Agent 基准。原库还记录部分数据波段元信息错误，说明 metadata 也需核验 |
| S9 | [GEOBench-VLM 官方仓库](https://github.com/The-AI-Alliance/GEO-Bench-VLM) / benchmark documentation | 31 个细分任务、8 类能力及 annotation/evaluation 说明可供对照；可借鉴 grounding/计数任务定义，不能直接把静态 VLM 分数当交互规划结果；本轮不转载其模型排名作为最新性能结论 |

本地证据：I1=`scripts/summarize_qwen_results.py` 与 runner/tests；I2=[memory 验收](evidence-memory-v1.md)；I3=[CloudSEN12 policy audit](cloud-policy-benchmark-v1.md)；I4=`config/eo-gym-source.json`；I5=[WHU 评价](whu-change-evaluation.md)；I6=[execution replay](execution-replay.md)。I2–I6 是项目已有记录，不能替代新一轮模型实验。

本轮访问锁文件中的 `https://github.com/paperuploadacount/EO-Gym` 得到 404；因此没有据此补写 EO-Gym 上游当前能力或许可结论。没有导入 Zotero、下载数据集或启动新模型服务。
