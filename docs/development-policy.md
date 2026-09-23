# 单一实现与 Git 版本管理规范

生效：2026-09-23。依据用户要求：代码不按 v1/v2 维护两套实现，重要阶段以 Git commit 保存。

## 日常约定

- 项目和当前设计统一称 EO Harness。新模块、配置、文档按职责命名，如 `core/`、`evaluation/`、`memory/`、`adapters/`，不因开发轮次新增 `v3/`、`system_v2.py` 或复制整套旧实现。
- 历史实现从 Git commit/tag 读取，不在活跃源码树保存备份副本。长期分叉的独立实现不再作为同步机制。
- A800 `/sata/yangm/eo-harness` 是实现/测试权威端。使用 `codex/<feature>` 短分支；重要可验收阶段做 Conventional Commit，完成后合并并按既有授权推送。
- 开工核对 local/A800/origin 的祖先关系，已公开历史不 rebase 或 force-push。A800 验收提交通过 Git 同步本地，避免各端重复实现、重复 commit 导致 SHA 分叉。
- 一个提交对应一个可验证变化：验收修复、模块迁移、任务包、文档分别提交，不为凑提交数拆分。
- 在干净且验收通过的提交上加 annotated tag（例如 `milestone/provenance-audit`）；tag 只是里程碑名称，不代替代码/模型/数据 hash。
- 不提交 dataset/model/runtime/artifact、SQLite/WAL、日志、证书/token、bulk metadata 或源数据 dump。只暂存明确文件并运行已有 payload guard。

## 发布版本与数据协议不同

Git 管理代码实现的历史。已经写入数据库、任务清单、trace、checkpoint 或被远端客户端使用的格式，还必须有可识别的契约标记；只知道服务 commit，不能自动判断任意历史记录的结构。

因此现有 `/v1`、`/v2` HTTP 路径、`v2_*` SQLite 表、schema/task/tool/evaluator 标记和内容 hash 暂作为**兼容边界**保留。后续不把它们作为复制业务实现的理由。既有 immutable manifest 不原位改名或改 hash；需要新结构时以迁移记录明确前后映射。

本次只建立规范与迁移计划，**尚未删除 `harness_api/app/v2`，尚未更名公开协议、数据库表和配置**。不能据本文声称代码已经没有版本命名。

## 有边界的迁移计划

| 当前形式 | 目标职责布局（待实现） | 兼容与验证要求 |
|---|---|---|
| `harness_api/app/v2/` 业务包 | `harness_api/app/core/` 及其 data/tools/renderer 子包 | 先移动实现、更新内部 imports；暂时的 import shim 只转发，不复制逻辑 |
| 同名 `domain.py/store.py/schemas.py` 与新业务包混置 | 当前 episode runtime 与 `compat/legacy_api/` 明确分离 | 活跃逻辑单一来源，旧 episode 只读投影与 HTTP fixtures 保持可用；有明确迁移方案后再退役兼容层 |
| 新类名带 `V2` | `EpisodeStore`、`EpisodeState` 等职责名 | 先区分 import 符号与 Pydantic/OpenAPI 序列化名；后者变化属于契约迁移，不能用全局替换自动批准 |
| `config/v2/`、新配置/文档名带开发轮次 | `config/environment/`、描述能力的文件名 | 所有 Compose、脚本、测试和文档引用同步；旧 reader 映射单独列出；数据任务版本号仍可保留 |
| runtime identity 仅记代码 commit | commit + image digest + model revision + task/input/tool/evaluator/adapter/seed pins | 复用已有 prospective identity 机制，补缺失项；新鲜执行重放与模型重新采样分别报告 |

迁移按小步独立 commit：imports/目录→兼容入口→配置引用→弃用清单；每步运行相关回归，最终检查历史 HTTP fixtures、旧 SQLite/任务读取、结构/执行 replay。无需为纯文档改动重跑 GPU 实验。

`scripts/check_git_payload.py` 拒绝路径中任意 `runtime` 段，因此源码目标使用
`core/`，而非 `app/runtime/`；不能为命名迁移放宽运行产物的 Git 防护。

## 活动文档维护

- [挑战与路线图](remote-sensing-challenges-and-roadmap.md) 是唯一活动改进清单；完成条目从该清单删掉，在对应专题文档留下结果/commit/证据路径。
- [当前架构](current-architecture-framework.md) 只讲现状，不堆历史 TODO。
- process notes 保存在用户指定的 `Codex-Work/EO-Harness`；不把过程记录重新放入源码目录。
- 旧文档可保留历史名称和历史结果，并添加当前结论入口；不改写当时原始实验产物。
