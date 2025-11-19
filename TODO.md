# TODO

## Backlog
- [ ] 汇总 QA 测试结果并反馈剩余问题
- [ ] 在 ip_layer.backward 中补全对 c/h 等参数的梯度（KKT 求导）
- [ ] 根据权威翼展或机型分级数据进一步细化 compat_matrix 规则
- [ ] 在更大规模实例（更多天+航班）上压测 GateIPFunction 与训练 runtime
- [ ] 系统性对比 mse_only / regret_only / combined 在整数 MILP 上的表现（多随机划分）
- [ ] 将压测与实验结果记录到简单的 CSV/日志文件中，便于后续分析
- [ ] 在更多种机场/机位配置上复现实验，观察方法泛化性
- [ ] 系统记录不同 γ / λ_regret 组合下的 integer regret 曲线
- [ ] 排查 midscale 可微 IP 的 Stage2 fallback 率过高问题，重点检查 big-M 与时间冲突约束的病态程度

## Current Cycle
- [ ] *（无）*

## In Progress
- [ ] 汇总 QA 测试结果并反馈剩余问题

## Done
- [x] 修复 train_relaxed / debug 脚本中 IPParams 初始化并完成 Gurobi 集成测试
- [x] 为训练稳定性在 relaxed 模型中实现时间/成本缩放
- [x] 静态审查 ip_layer.py 的 HSD/KKT 实现是否符合规范
- [x] 运行 debug_ip_layer 验证 HSD 求解器与 HiGHS 对比（精度/状态）
- [x] 运行 debug_ip_grad 检查解析梯度与数值梯度一致性
- [x] 深入审查 src/gate_assignment/ip_layer.py 现有实现与 qpth 依赖，整理替换需求
- [x] 设计并实现 numpy 版 HSD 内点前向求解器（solve_lp_numpy 等）
- [x] 推导并编码 KKT backward（solve_kkt_backward）用于梯度传递
- [x] 重写 GateIPFunction/gate_ip_solve，移除 qpth 并接入新 solver
- [x] 在最小示例上运行 sanity check，验证 forward/backward 基本正确
- [x] 设计中等规模实验参数（航班数、机位上限、实例数、模式）
- [x] 扩展 run_experiments_two_stage 支持 midscale 实验（独立 save_dir、CSV 不覆盖）
- [x] 运行 midscale 实验，对比 mse_only vs combined 的整数 regret 与 fallback 比例
- [x] 新建 real-scale 评估脚本，针对 119 机位 + 大航班数做整数 Two-Stage 评估
- [x] 跑一组 real-scale 评估实验，保存到新的目录并撰写实验总结
- [x] 固定 ip_layer.py 的公共接口并确保 backward 形状正确
- [x] 新建 debug_ip_layer.py 验证 forward/backward
- [x] 阅读 NSP IPOfunc.backward，弄清 KKT 梯度如何映射到 h
- [x] 在 ip_layer.py 中实现内点 forward + KKT backward（仅对 h）
- [x] 编写 debug_ip_grad.py，使用有限差分验证 GateIPFunction 对 h 的梯度
- [x] 调整 LP 结构，使到达时间只出现在约束 RHS h
- [x] 在 model_relaxed.py 中新增基于 gate_ip_solve 的 torch 版 Stage1/Stage2 relaxed 接口
- [x] 修改 train_relaxed.py，让 regret 完全在 torch 计算图中构建
- [x] 在小规模实例上验证训练循环（loss/MSE/regret 正常且可反传）
- [x] 在更大规模实例上压测 GateIPFunction（runtime & inaccurate solution 统计）
- [x] 实验脚本：自动训练 + 整数评估 + 汇总结果，并记录 CSV
- [x] 实现数据集划分（train/val/test），按日期/实例划分
- [x] 扩展 train_relaxed.py 支持不同 loss 配置并保存 checkpoint
- [x] 新建整数评估脚本，使用整数 Two-Stage MILP 在 test 集上评估模型
- [x] 跑通一次小规模实验，对比至少两种 loss 配置下的整数 regret 指标

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
