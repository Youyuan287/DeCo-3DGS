# ARIS ReLaGS 项目配置

当前状态：**2026-10-24 HF-only基线输出已核验，64.0068%；等待方法决策，不自动启动GPU。** 详情见STATE_20261024.md。自建lerf_mask未验证；不改共享环境、不重复打补丁、不删数据。

ARIS是技能工作流集合。本包用项目`AGENTS.md`/`CLAUDE.md`、可机器读取的约束清单和两个API建议角色配置它；没有假装安装一个统一ARIS后台，也没有改全局Codex/Claude模型或创建不存在的MCP服务。

- 服务器：`ssh -i ~/.ssh/id_ed25519 -p 30253 root@fj01-ssh.gpuhome.cc`
- 远端配置工作目录：`/root/rivermind-data/ARIS-ReLaGS`
- 规划角色：`deepseek-v4.1-flash-1017`
- 审查角色：`glm-5.3-prime`
- 中转API：`https://maas.qianwenaiapi.com/compatible-mode/v1`
- 当前执行端负责实际代码；两个API角色只给建议，不共享写权限。

启动研究时，在本目录打开带有既有ARIS技能的执行端，使其读取AGENTS.md/CLAUDE.md。本包不在远端安装Codex/Claude CLI，不触发完整研究循环。HF输出已接受；选定一个改动并固定实际代码入口后，再按用户授权运行实验。

配置文件：`.aris/project.json`保存路径、待验收基线、评测口径、磁盘和执行限制；`.aris/api_roles.json`保存模型名/角色prompt，没有密钥。私有密钥仅远端`.aris/credentials.json`，目录700、文件600，加入ignore；也支持环境变量覆盖。禁止将此私有文件复制到Git或报告。已出现在聊天中的密钥建议之后轮换。

只测试API（不会运行ReLaGS）：

```bash
cd /root/rivermind-data/ARIS-ReLaGS
python3 scripts/llm_bridge.py --role planner --check
python3 scripts/llm_bridge.py --role reviewer --check
```

后续获准发起研究审查时，可将不含密钥的研究简报和必要证据写入一个prompt文件：

```bash
python3 scripts/llm_bridge.py --role planner --prompt-file RESEARCH_BRIEF.md --output review-stage/planner_001.json
# 先由执行端写reviewer输入，包含方案与必要证据，不让模型自行执行。
python3 scripts/llm_bridge.py --role reviewer --prompt-file review-stage/reviewer_input_001.md --output review-stage/reviewer_001.json
```

脚本不会自动读取整个仓库上传API；只发送指定prompt文件。本次部署只做各一次最小连通测试，不发送完整研究报告。已有同名输出拒绝覆盖。未来实验使用screen名`aris-relags`，现在不创建空会话冒充运行中。

运行配置包自检：`python3 -m unittest discover -s tests -v`。本配置的GPU等待状态属于工作流约束，不是对其他会话或系统的强制沙箱。不要执行历史run_hf_only.sh覆盖results。所有模型/数据和当前未跑通的ReLaGS代码保留。
