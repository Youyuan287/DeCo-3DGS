# 配置部署结果

2026-10-23部署至`/root/rivermind-data/ARIS-ReLaGS`；服务器`fj01-ssh.gpuhome.cc:30253`，只读核验hostname为`jupyter-ijj2vk1r1mehp255`。连接身份`id_ed25519`，未变更任何SSH授权。

| 项目 | 已验证状态 |
|---|---|
| planner | deepseek-v4.1-flash-1017，指定DeepSeek密钥，最小请求返回OK |
| reviewer | glm-5.3-prime，指定GLM密钥，最小请求返回OK |
| API | https://maas.qianwenaiapi.com/compatible-mode/v1 |
| 请求内容 | 各一次仅要求回答OK；未上传研究报告/代码/数据 |
| 本地配置/桥接自检 | 5项通过 |
| 远端Linux自检 | 5项通过，含600权限检查 |
| 私有文件 | 仅远端`.aris/credentials.json`，权限600；目录700；未复制到本地或版本库 |
| 当前研究/实验 | 未启动；未创建screen；未修改ReLaGS代码 |
| 当前基线 | WAITING_FOR_USER_RELAGS_READY，accepted=false |

API连通记录：远端`logs/planner_connection_20261023.json`、`logs/reviewer_connection_20261023.json`。最小请求只验证模型ID/凭据/接口能响应，不能证明模型研究能力或长期可用性。

远端两个现存ReLaGS目录不能混用。当前用户明确尚未跑通；既有历史64.01%的HF-PT产物不构成当前baseline验收。等用户通知后确认实际工作目录、环境及产物，再作方法研究与实验决策。

配置包位置即工作目录；AGENTS.md/CLAUDE.md用于现有ARIS执行端读取。没有安装新的全局CLI/MCP，没有修改全局Codex/Claude配置。GPU停止状态是工作流约束，并非系统级强制沙箱。

已修复的配置包验证问题：Windows沙箱默认临时目录不允许写入，测试改用配置包内独立临时目录；证据复制首次工作目录解析失败，随后用绝对路径重做并确认两份快照存在。最终本地和远端测试均通过。未修复任何当前ReLaGS运行问题。

密钥轮换：以后可优先通过ARIS_DEEPSEEK_API_KEY与ARIS_GLM_API_KEY环境变量覆盖；私有文件安装脚本拒绝覆盖已存在文件。不要把密钥再贴进报告或命令参数。聊天中已出现的密钥建议适时轮换。
