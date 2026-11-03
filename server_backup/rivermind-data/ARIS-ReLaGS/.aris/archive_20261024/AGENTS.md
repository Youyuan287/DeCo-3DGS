# ARIS — ReLaGS 项目指令

先读取 `.aris/project.json`、`RESEARCH_BRIEF.md`、`evidence/research_audit.md`和`evidence/experiment_evidence.csv`。这些是本项目上下文，不是批准运行实验的凭证。

## 当前停点

用户2026-10-23明确表示：当前服务器ReLaGS尚未跑通，跑通后会通知。等待该通知。不得自动排障/修复ReLaGS、重跑基线、启动GPU、训练、SAM、提特征、建图或参数扫描。历史64.01只是官方PT路线的参考，不是当前实验验收。新克隆`ReLaGS-1023`与历史`ReLaGS_Reproduction`不得混用。

## 目标与角色

目标是在实际验收的ReLaGS上做出一个清楚且可持续改进的核心方法，并提升真实最终分割指标。研究审计里的空间支持核验只是首个待证伪假设，不是预设必须成功的答案。

- 当前Codex/本地执行端：证据发现、实现和运行的唯一负责人。
- DeepSeek planner：通过`python scripts/llm_bridge.py --role planner --prompt-file <文件>`给出方法方案。
- GLM reviewer：通过相同桥接脚本的`--role reviewer`作独立审查。
- 两个API角色不执行shell、不修改代码、不自动互相调用。回答是建议，不是用户授权。

使用现有ARIS技能时优先读本项目约束。通用`research-pipeline`、`idea-discovery`、`auto-review-loop`默认的自动pilot、四轮循环和自动推进在本项目均关闭。通用LLM审查使用本桥接器或`auto-review-loop-llm`，不要误触发只支持OpenAI审稿人的`research-review`默认模型分支。

## 用户通知跑通以后

先确定用户当前真正运行的目录、环境、命令和产物，不默认就是历史目录；在用户授权范围内核验并记录baseline及输入/源码哈希。禁止直接执行历史`run_hf_only.sh`覆盖`results/`。新输出必须放数据盘独立目录，原仓库已有修改保留。

用历史证据约束方法凝练：最多三个竞争方案，选择一个；每次只改一个算法对象；先冻结候选/评测做最小配对。方法和参数冻结后才考虑扩场景、多seed与完整消融。不要加一串新的Stage编号。缺原始缓存时先列缺项，不借新seed或另一wave冒充原状态。

严禁GT参与触发/候选构建/推理选择，严禁按场景名选最好配置，严禁用oracle、2D补洞、subset proxy替代正式3D成绩。历史反例包括纯校准不足、local-contrast退化、统一NMS/扩k退化、Alpha非整体/部件双专家、增强全栈不优于统一pure。相同mIoU不等于代码没生效；须核查特征、候选ID和Gaussian并集。

每个被批准的运行保存命令、环境变量（密钥脱敏）、源码/模型/NAG哈希、逐query指标、选中节点、空输出变化和统计分母。正式主口径：图内query平均、场景内图像平均、四场景平均。双方非空的条件均值与对总体增益的贡献分别记录。未来GPU最多1作业；数据盘预留10GiB、系统盘5GiB，不满足则先报告，不自动删除旧模型/数据。

所有API密钥只从环境或`.aris/credentials.json`读取。不得读取后打印，不放prompt、报告、Git或运行命令参数。日志只保存请求角色/模型、必要prompt和完整模型回复，不保存Authorization。无endpoint跨站重定向。密钥已在聊天中出现，应支持轮换。
