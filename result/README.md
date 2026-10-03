# 运行产物

- `mvtec_full/`：384 张真实图像的掩膜、测量叠加图和 JSON。
- `mvtec_evaluation/`：全量定位报告、逐图匹配记录和失败缩略图。
- `comparison_frozen/`：冻结参数下 Otsu、自适应阈值、Lab 的代表样例对比。
- `synthetic/`、`synthetic_evaluation/`：合成像素测量、公差与尺寸真值比较。
- `synthetic_mm/`、`synthetic_mm_evaluation/`：合成尺度换算和毫米流程验证。
- `edge_cases/`：坏图、无目标、螺母、粘连和截断的结果。
- `validation.log`：`tools/run_validation.py` 的完整执行日志。

这些是可再生成的本地运行产物。`comparison/` 为参数探索中间结果，正式对比使用 `comparison_frozen/`；不据中间结果声称最终性能。
