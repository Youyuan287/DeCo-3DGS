# ARIS项目配置产物索引

| 日期 | 文件 | 用途 |
|---|---|---|
| 2026-10-23 | .aris/project.json | 待验收ReLaGS、评测参考、研究约束和禁止自动实验 |
| 2026-10-23 | .aris/api_roles.json | DeepSeek规划与GLM审查角色，不含密钥 |
| 2026-10-23 | AGENTS.md / CLAUDE.md | ARIS执行端项目指令 |
| 2026-10-23 | RESEARCH_BRIEF.md | 研究目标、历史证据与当前等待点 |
| 2026-10-23 | evidence/research_audit.md / experiment_evidence.csv | 前轮审计及32项证据快照 |
| 2026-10-23 | scripts/llm_bridge.py | 无GPU、无shell执行能力的API建议桥接器 |
| 2026-10-23 | scripts/install_credentials.py | 标准输入安装私有凭据，不回显值 |
| 2026-10-23 | tests/test_llm_bridge.py | 端点/凭据/请求/配置验证 |
| 2026-10-23 | DEPLOYMENT_20261023.md | 本次部署与验证记录 |
| 2026-10-24 | STATE_20261024.md / .aris/baseline_verified_20261024.json | 当前HF输出验收与208条结果/源码模型哈希；GPU仍未授权启动 |
| 2026-10-24 | scripts/verify_baseline.py | 只读CPU核验，不导入模型/renderer |

未把任何秘密文件列作交付资料；远端credentials受ignore及权限保护。没有生成或声称任何新实验指标。
