# data directory

```
data/
├── raw/        # 原始输入：航班、机位、滑行距离、兼容矩阵等
└── clean/      # 由 gate_assignment.data.preprocess 生成的清洗/补全版本
```

- `raw/` 内保留供应商提供的 CSV，不做任何改动，作为追溯与再生成的基准。
- `clean/` 包含：
  - `t3cde_candidate_flights_final_v2.backup.csv`：建模用航班基准。已剔除没有滑行距离的 928 条航班（落地机位为 `A113`、`Mxx`、`N10x/N205`、`W2xx`），并删除 `takeoff_bay`、`出港跑道` 等与 T3 优化无关的列，只保留进港侧所需字段；
  - `T3CDE_stands_clean.csv`：标准化后的 T3C/D/E 机位清单。缺乏距离的机位（371–373、381–382、402、404、412、537–550）已移除，剩余 119 个机位均能在距离表中查到滑行距离；
  - `PEK_stand_runway_directional_taxi_distances.csv`：包含 291 个机位到 18L/36R/01/19/18R/36L 跑道的滑行距离矩阵，是所有滑行距离的唯一来源；
  - `PEK_complete_stand_runway_distances.csv`（如存在）：从 OSM/Overpass 提取的扩展机位距离，仍缺少个别西侧机位，可作为补充参考。

如需重新生成 `clean/` 内容，运行：

```bash
cd /Users/wp/code/11.14
conda run -n gate_assignment python -m gate_assignment.data.preprocess
```

脚本将读取 `raw/` 中的文件，产出新的清洗结果并打印插补统计。*** End Patch
