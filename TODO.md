# TODO

## Backlog
- [ ] 在 ip_layer.backward 中补全对 c/h 等参数的梯度（KKT 求导）
- [ ] 将两阶段 LP 松弛接口接入后续神经网络训练流程
- [ ] 根据权威翼展或机型分级数据进一步细化 compat_matrix 规则
- [ ] 在 GateIPFunction.backward 中实现对 h 的真实梯度，移除 numpy regret detour
- [ ] 在 model_relaxed.py 中新增 torch 版 Stage1/Stage2 relaxed 接口
- [ ] 修改 train_relaxed.py，让 regret 完全在 torch 计算图中构建

## In Progress
- [ ] *（无）*

## Done
- [x] 固定 ip_layer.py 的公共接口并确保 backward 形状正确
- [x] 新建 debug_ip_layer.py 验证 forward/backward

## Done
- [x] 在 GateAssignmentInstance 中补充计划落地时间等特征，构造训练特征矩阵
- [x] 新建 ArrivalPredictor（PyTorch）并接入训练
- [x] 实现 GateAssignmentDataset，将实例打包为 Dataset
- [x] 编写 train_relaxed.py：Two-Stage relaxed + MSE 训练骨架
- [x] 编写 debug_relaxed_two_stage.py，对比整数两阶段与 relaxed 两阶段
- [x] 封装两阶段 LP 松弛求解接口（Stage1/Stage2）
- [x] 在 model_relaxed.py 中实现 Stage2 LP 松弛矩阵构造（含 |x - x1| 惩罚）
- [x] 编写 debug_relaxed_stage1.py，对比 Gurobi LP 与内点解
- [x] 设计并实现 GateAssignment Stage1 LP 松弛的矩阵构造函数
- [x] 在 src/gate_assignment/ 下复制并精简可微 IP 层，实现 ip_layer.py
- [x] 阅读 reference/code/NSP/ip_model_whole.py，理解 IPOfunc 接口与矩阵格式
- [x] 根据机型/翼展等规则完善 compat_matrix 构造逻辑
- [x] 优化时间冲突建模，仅为潜在冲突的机位-航班组合创建排序变量
- [x] 编写 debug_two_stage.py，运行两阶段流程并计算 post-hoc regret
- [x] 实现 Stage2 MILP：使用真实到达时间 + 对 x 与 Stage1 解差异的惩罚 |x - x1|
- [x] 实现 Stage1 MILP：使用预测到达时间求解，目标仅为总滑行距离
- [x] 在 model_mip.py 中拆分/扩展接口，支持“给定任意到达时间向量”的通用 MILP 构造
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
