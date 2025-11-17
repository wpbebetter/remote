# AGENTS.md

本文件约束本仓库中 **所有 AI 辅助开发流程** 的行为规范。  
目标：在复现《Two-stage predict+optimize for MILPs》方法的同时，将其迁移到“PEK T3CDE 机位分配 + 滑行距离最小化”的应用中。

---

## 0. 总体原则

- 一切操作以本仓库和本文件为准，不擅自发明目录和流程。
- 任何时候，**先思考，再动手**；先写 TODO，再写代码。
- 所有与用户的自然语言交流默认使用 **简体中文**。
- 严格遵守用户的显式指令；如与本文件冲突，以用户最新指令为准。

---

## 1. 仓库结构约定

当前关键结构（可能随时间扩展）：

- `data/clean/`
  - `PEK_stand_runway_directional_taxi_distances.csv`
  - `t3cde_candidate_flights_final_v2.backup.csv`
  - `T3CDE_stands_clean.csv`
- `data/raw/`  
  原始数据，仅读取不修改。
- `reference/`  
  - `code/0-1 knapsack`
  - `code/Alloy production`
  - `code/NSP`  
  **官方参考代码，只能阅读和学习，不得修改或在其中添加新代码。**
- 未来由你（AI）创建的新目录（建议）：
  - `src/`：实际可运行的机位分配与训练代码。
  - `src/gate_assignment/`：机位分配 MILP + 两阶段模型。
  - `notebooks/`：探索性分析（可选）。
  - `TODO.md`：全局任务清单（见下文）。

要求：

- **所有新代码、新脚本、新配置文件一律放在 `src/` 或其子目录**。  
  绝对不要把任何新文件写进 `reference/`。
- 读取数据时使用 `data/clean`，如需生成中间文件，写入 `data/processed`（如不存在请先创建）。

---

## 2. 沟通与输出规范

- 对用户的回答统一使用 **简体中文**，除非用户明确要求使用其他语言。
- 解释概念时：
  - 先给直白解释，再给公式/伪代码；
  - 避免无用的客套和水话，优先给出可执行的信息。
- 如用户让你“写代码”“写配置”，在回复中：
  - 使用 Markdown 代码块；
  - 给出最小可执行示例，并说明文件应存放的路径。

---

## 3. 工作流与 TODO 机制

在每次收到任务（包括用户消息或新问题）后，必须按以下顺序行动：

1. **思考当前任务**：
   - 明确输入、输出、相关文件夹。
2. **更新根目录下的 `TODO.md` 文件**：
   - 若不存在，则创建，基础结构如下：
     ```md
     # TODO

     ## Backlog
     - [ ] 任务说明...

     ## In Progress
     - [ ] 任务说明...

     ## Done
     - [ ] 任务说明...
     ```
   - 将本次要完成的子任务拆成 2–6 条原子项，写入 `Backlog`。
   - 把正在处理的任务移动到 `In Progress`。
3. **实施任务**：
   - 按 TODO 顺序逐项完成；
   - 完成后，将对应条目从 `In Progress` 移到 `Done`。
4. **在回复中简要同步**：说明本轮新增或完成了哪些 TODO 项（不必全文复制，但要清楚指出变动）。

---

## 4. 开发环境和依赖

### 4.1 Python & 包管理

- 使用 **Python 3.10+**，统一在 `conda` 的 `base` 环境中工作，不再创建额外 `venv` 或自定义环境；执行脚本前 `conda activate base`，确保 `python` 与 `pip` 指向同一解释器。
- 本项目的所有优化求解统一依赖 `gurobipy`（Gurobi 官方 Python 接口），禁止切换到其它 MILP/LP 求解器。
- 依赖统一写入根目录 `requirements.txt`，核心包包含 `numpy`、`pandas`、`pytorch`、`gurobipy`、`scipy` 等。
- 添加依赖步骤：
  1. 在 `requirements.txt` 中追加新包；
  2. 在已激活的 `base` 环境中执行 `pip install -r requirements.txt`；
  3. 运行一次最小脚本（如 `python -c "import gurobipy"`）确认无 ImportError。

### 4.2 运行与测试

- 所有脚本应设计为可以在项目根目录下运行，例如：
  - `python -m src.gate_assignment.train`
- 为核心模块编写最少量的单元测试：
  - 测试路径：`tests/`，例如 `tests/test_gate_assignment_model.py`。
  - 使用 `pytest`，执行：`pytest -q`。
- 在每次重要修改后，必须确保：
  - 核心脚本可以运行完一个小实例（例如仅一日航班）；  
  - 所有测试通过。

---

## 5. 参考代码使用规范

- `reference/` 目录 **只做学习参数化建模与训练框架之用**：
  - 允许操作：阅读、复制片段到 `src/` 并适当修改；  
  - 禁止操作：在 `reference/` 内创建、编辑、删除任何文件。
- 当你“不知道怎么写代码”时，遵循顺序：
  1. 在 `reference/code/Alloy production` 中查看：  
     - 如何将参数写到约束的矩阵形式；
     - 如何实现两阶段问题。
  2. 在 `reference/code/NSP` 中查看：  
     - 参数在 RHS（h 向量）中的用法，与本项目最相近。
  3. 将理解转化为 `src/gate_assignment` 下的实现，而不是直接复制粘贴整文件。

---

## 6. 机位分配项目的专用约定

为了确保所有 AI agent 写出的代码结构一致，必须遵守以下模块划分，并在涉及求解器的部分统一调用 `gurobipy`：

1. `src/gate_assignment/data.py`
   - 负责读取 `data/clean` 下三个 CSV；
   - 构建：
     - 每日航班实例；
     - 滑行距离矩阵；
     - 机位兼容矩阵。

2. `src/gate_assignment/model_mip.py`
   - 实现 **单阶段 MILP**：
     - 决策变量：机位分配及时间冲突辅助变量；
     - 目标：总滑行距离；
     - 约束：唯一分配、兼容、时间不冲突。
   - 所有模型均通过 `gurobipy` 建立和求解，并在此基础上扩展为 **两阶段 Stage 1 / Stage 2 模型**。

3. `src/gate_assignment/model_relaxed.py`
   - 将 MILP 转为 barrier LP 松弛（参考 NSP 示例）；
   - 对接内点 + KKT 求导模块，仍然以 `gurobipy` 求得参考基准；
   - 提供接口：`solve_stage1_relaxed(theta_hat)` / `solve_stage2_relaxed(...)`。

4. `src/gate_assignment/net.py`
   - 定义预测到达时间的神经网络结构（PyTorch）。

5. `src/gate_assignment/train.py`
   - 训练入口：  
     - 加载数据 → 网络预测 → 两阶段松弛求解 → 计算 post-hoc regret → 反向传播。

6. `src/gate_assignment/eval.py`
   - 评估脚本：  
     - 使用整数模型（Gurobi MILP）求真 Stage1 / Stage2 / 真最优；  
     - 计算并输出各项指标。

所有 agent 在实现新功能时，应尽量往上述文件中补充，而不是另起无关命名。

---

## 7. 代码风格与质量

- 语言：Python 代码遵循 PEP 8，变量命名清晰（如 `arrival_time_pred`, `taxi_distance_matrix`）。
- 所有核心函数必须写 **docstring**，描述：
  - 参数、返回值；
  - 函数的数学含义（特别是与论文中符号的对应）。
- 尽量避免神奇数字（magic numbers），例如 big-M、惩罚系数，应定义成常量，带有注释。
- 任何复杂约束（尤其时间冲突与 penalty 线性化部分）必须在代码注释中标出对应论文章节或公式。

---

## 8. Git 工作流

每次完成一组清晰的改动后，按以下流程操作：

1. 确保所有脚本与测试可以运行：
   - `pytest -q`（若已有测试）；
   - 至少跑通一个小规模实例。
2. 查看 diff：
   - `git status`
   - `git diff`
3. 添加文件：
   - `git add <files>` 或 `git add .`（慎用）。
4. 写清晰的 commit message，格式建议：
   - `[gate] 描述本次改动`  
     例如：`[gate] add basic MILP model for gate assignment`
5. 提交：
   - `git commit -m "<message>"`
6. 推送：
   - `git push`

禁止事项：

- 禁止提交明显不能运行的代码（语法错误、缺失依赖、路径错乱）。
- 禁止把大文件（>50MB）随意加入 Git，尤其是原始数据和模型 checkpoint。

---

## 9. 遇到问题时的处理流程

当你（AI）“不会写 / 不确定”时，必须遵循以下顺序，而不是胡乱尝试：

1. 在 `reference/code` 下找到最相近的示例，阅读对应实现。  
2. 在 `src/` 下创建一个 **草稿脚本**（例如 `scratch.py` 或 notebook）做小规模试验。  
3. 确认逻辑正确后，再将代码整理进正式模块。  
4. 如仍有不确定点：
   - 在 `TODO.md` 中加入“需要人工确认”的条目；
   - 在对用户的回复中明确指出不确定之处与已尝试的方案。

---

## 10. 最后提醒

- 所有 agent 必须把自己当成“严谨的合作者”：  
  - 不隐藏失败尝试；  
  - 不伪造结果；  
  - 不随意改动 `reference/` 与数据文件。
- 若用户明确给出新的高优先级指令（例如改变目标函数、文件结构），应：
  1. 先在 `TODO.md` 中新增/调整条目；
  2. 再调整本文件中相关说明（仅在确需时修改 AGENTS.md）。

> 简单说：**读论文→看 reference→写 TODO→在 src 里实现→跑测试→git 提交→用中文汇报**。  
> 任何时候都不要在 `reference` 里写代码，也不要跳过思考和 TODO 步骤。
