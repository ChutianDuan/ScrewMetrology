# 运行产物

- `mvtec_full/`：384 张真实图像的掩膜、测量叠加图和 JSON。
- `mvtec_evaluation/`：全量定位报告、逐图匹配记录和失败缩略图。
- `comparison_frozen/`：冻结参数下 Otsu、自适应阈值、Lab 的代表样例对比。
- `axis_comparison/`：最新的固定 mask 与候选的 PCA／fitLine／旋转框对照，含方向与尺寸误差、覆盖率、失败案例和并排图；复现方式见 [主轴对比文档](../docs/axis_comparison.md)。
- `synthetic/`、`synthetic_evaluation/`：合成像素测量、公差与尺寸真值比较。
- `synthetic_mm/`、`synthetic_mm_evaluation/`：合成尺度换算和毫米流程验证。
- `edge_cases/`：坏图、无目标、螺母、粘连和截断的结果。
- `validation.log`：`tools/run_validation.py` 的完整执行日志。

这些是可再生成的本地运行产物，只有本说明文档纳入 Git。旧调参目录 `comparison/` 及重复主轴报告 `axis_comparison_20261005/`、`axis_comparison_20261005_final/` 已清理；最新 `axis_comparison/` 和历史验证目录保留。正式分割对比使用 `comparison_frozen/`，不据调参中间结果声称最终性能。
