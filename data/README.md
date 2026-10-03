# 本地数据

原始图像、下载包、旧项目数据和合成产物不提交到 Git；已有文件仍保留在本地。

从项目根目录获取 MVTec Screws：

```sh
python3 tools/download_dataset.py
```

官方来源：[MVTec Screws](https://www.mvtec.com/research-teaching/datasets/mvtec-screws)，作者为 MVTec Software GmbH。下载脚本记录来源和校验值，解压后保留原包说明。数据集采用 CC BY-NC-SA 4.0 许可，使用时遵循包内说明；项目文档中的真实图像示例也来自该数据集。

构建后生成学习用合成图像：

```sh
./build/generate_samples data/synthetic
```

完整构建、测量和评价步骤见项目根目录的 README。
