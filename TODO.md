# TODO

## Backlog
- [ ] 根据机型/翼展等规则完善 compat_matrix 构造逻辑
- [ ] 优化时间冲突建模，仅为潜在冲突的机位-航班组合创建排序变量

## In Progress
- [ ] *（无）*

## Done
- [x] 在 src/gate_assignment/debug_mip.py 中写一个小脚本，选取一天数据运行 MILP 并打印结果
- [x] 创建 src/gate_assignment/model_mip.py 并实现单阶段机位分配 MILP（仅使用真实到达时间）
- [x] 在 data.py 中实现从三份 CSV 构造每日 GateAssignmentInstance 的函数
- [x] 设计 GateAssignmentInstance 数据结构，封装每日航班与机位信息
- [x] 阅读 reference/code/NSP 示例，理解可微 IP 层的接口
- [x] 创建 src/gate_assignment/ 目录与基础文件
- [x] 在 src/gate_assignment/data.py 中实现三个 CSV 的读取与基本检查
- [x] 添加一个简单的调试脚本，验证数据加载无误
- [x] 阅读 AGENTS.md 和 README.md，确认规范
- [x] 重建 TODO 文件结构并记录本轮任务
- [x] 汇总用户提供的新规范并整理成可写入 AGENTS.md 的结构
- [x] 将最新规范完整合并进 AGENTS.md
- [x] 自查 AGENTS.md 字数、格式与语言要求
- [x] 在最终回复中同步 TODO 变动与校验结果
- [x] 调整 AGENTS.md 中 Python/环境描述以符合 base conda 要求
- [x] 强调 gurobipy 为唯一优化器选择并更新相关段落
- [x] 在修改完成后复查字数与格式
- [x] 历史任务记录：
    1. [x] 更新 AGENTS.md 文档及其所有子项
    2. [x] 二次修订 AGENTS.md 使内容与实际结构匹配
