# 楼道适老化改造的阶段效果与续建判断

区里收集楼道设施与出行机会的周资料。所有日期是北京时间，区间左闭右开，每周从周一开始。出行机会指住户当周确实使用小区出入口的次数。moved 指不再属于该住址的观察总体。设计范围和实际通过路线分开。

资料均为合成记录，采用 UTF-8 编码。交换字段、标识和状态见[资料说明](docs/data.md)；口径与结论见[方法与结果](docs/analysis.md)。

项目使用 Python 3.11.2，资料查看与检测只依赖标准库。在项目目录运行：

```sh
python3 inspect_records.py                          # 记录数与字段检查
python3 -m unittest discover -s tests -v            # 交换约定与分析回归测试
python3 -m analysis.report --out outputs            # 生成报告与可复核台账
```

`python3 -m analysis.report` 会输出：

- `outputs/report.md`：阶段效果、对照重核、施工与间断、续建审查、会议回查索引；
- `outputs/availability_weeks.csv`：每个楼栋-楼层-周的可用组合、状态、依据阶段与同期其他项目；
- `outputs/estimates.csv`：每条估计的窗口、样本量、差中差、区间、缺报边界与结论强度。

同一份资料重复运行得到逐字节相同的输出；所有数字都可由 `analysis/` 重算。
