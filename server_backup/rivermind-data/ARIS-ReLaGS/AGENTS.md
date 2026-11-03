# ARIS — ReLaGS 项目指令

## 最新用户裁决（覆盖下方旧停点，2026-10-24）

本次OVS四组已完成：63.2871/49.4952/52.7068/16.6072（baseline/visibility/spatial/permutation），56query各组，当前空间乘法规则不成立。停止扩场景和扫描，不自动追加方法；详见review-stage/spatial_support_20261024/OVS_RESULTS.md及远程runs/ovs_support_20261024。原始分数重评分会同时改变排序和gap截断基数，不能再把所有退化称排名错误。pirate hat支持不重合的数值已记录，但投影mask独立可视化复核仍缺，不先断定语义误认。

主任务必须是LERF-OVS，不能混入LERF-Mask成绩。用户已批准figurines固定候选四组配对及向指定API发送研究摘要/非敏感路径/代码供DeepSeek与GLM审查。范围仅限缓存复用和推理，不授权训练、SAM、重提特征、建图或扫参。全局自动GPU权限仍关闭；本次按OVS_EXPERIMENT_PLAN.md的具体授权执行。LERF-Mask73.13%只作旁支核验，不能算OVS进展。当前可追溯OVS模型是作者PT，明确标注；不冒称自建PT。具体进度以review-stage/spatial_support_20261024下记录及远程runs/ovs_support_20261024产物为准。

先读取 `.aris/project.json`、`RESEARCH_BRIEF.md`、`evidence/research_audit.md`和`evidence/experiment_evidence.csv`。这些是本项目上下文，不是批准运行实验的凭证。

## 当前停点

用户2026-10-24已通知迁移与HF-only复现完成。本轮CPU核验ReLaGS_run_20261023的208条query、四场景summary及原历史逐query结果一致，宏平均64.0068369583%。HF输出基线已接受，不等于自建lerf_mask/从零语义场已通过。当前等待方法路线决策；不自动启动GPU、训练、SAM、提特征、建图或参数扫描，不删除任何模型和数据。两份evaluator的内容哈希不同，实际方法入口必须先固定；不得混用ReLaGS-1023与ReLaGS_Reproduction。

## 目标与角色

目标是在实际验收的ReLaGS上做出一个清楚且可持续改进的核心方法，并提升真实最终分割指标。研究审计里的空间支持核验只是首个待证伪假设，不是预设必须成功的答案。

- 当前Codex/本地执行端：证据发现、实现和运行的唯一负责人。
- DeepSeek planner：通过`python scripts/llm_bridge.py --role planner --prompt-file <文件>`给出方法方案。
- GLM reviewer：通过相同桥接脚本的`--role reviewer`作独立审查。
- 两个API角色不执行shell、不修改代码、不自动互相调用。回答是建议，不是用户授权。

使用现有ARIS技能时优先读本项目约束。通用`research-pipeline`、`idea-discovery`、`auto-review-loop`默认的自动pilot、四轮循环和自动推进在本项目均关闭。通用LLM审查使用本桥接器或`auto-review-loop-llm`，不要误触发只支持OpenAI审稿人的`research-review`默认模型分支。

## 用户通知跑通以后

已完成HF结果核验，见STATE_20261024.md和.aris/baseline_verified_20261024.json。进入方法实验前还需固定执行仓库、评测脚本和命令；不把当前哈希当原运行来源的绝对证明。禁止直接执行历史run_hf_only.sh覆盖results。新输出放数据盘独立目录；共享thgs conda环境不得改动，绝对软链接需保持；现有兼容补丁不得重复应用。

用历史证据约束方法凝练：最多三个竞争方案，选择一个；每次只改一个算法对象；先冻结候选/评测做最小配对。方法和参数冻结后才考虑扩场景、多seed与完整消融。不要加一串新的Stage编号。缺原始缓存时先列缺项，不借新seed或另一wave冒充原状态。

严禁GT参与触发/候选构建/推理选择，严禁按场景名选最好配置，严禁用oracle、2D补洞、subset proxy替代正式3D成绩。历史反例包括纯校准不足、local-contrast退化、统一NMS/扩k退化、Alpha非整体/部件双专家、增强全栈不优于统一pure。相同mIoU不等于代码没生效；须核查特征、候选ID和Gaussian并集。

每个被批准的运行保存命令、环境变量（密钥脱敏）、源码/模型/NAG哈希、逐query指标、选中节点、空输出变化和统计分母。正式主口径：图内query平均、场景内图像平均、四场景平均。双方非空的条件均值与对总体增益的贡献分别记录。未来GPU最多1作业；数据盘预留10GiB、系统盘5GiB，不满足则先报告，不自动删除旧模型/数据。

所有API密钥只从环境或`.aris/credentials.json`读取。不得读取后打印，不放prompt、报告、Git或运行命令参数。日志只保存请求角色/模型、必要prompt和完整模型回复，不保存Authorization。无endpoint跨站重定向。密钥已在聊天中出现，应支持轮换。
