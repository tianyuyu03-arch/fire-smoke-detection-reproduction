# APConv / AEL-YOLO 真实复现工作目录

## 内容

- `code/Finaldata/`：作者公开的模型模块、YAML、训练和验证脚本（Git commit `5b3588189d3a7e6776b06492f8ba5c037c722e3c`）。
- `code/ultralytics-8.3.40/`：Ultralytics v8.3.40 基础代码；已把作者的 `nn/tasks.py` 和三个自定义模块接入，原版 `tasks.py` 备份在 `notes/tasks.py.upstream-8.3.40`。
- `dataset/finaldata.zip`：作者 Google Drive 发布的原始压缩包。
- `dataset/finaldata/`：解压后的 Finaldata，使用作者提供的 train/valid 切分。
- `dataset/dataset.yaml`：本机数据路径，类别 ID 为 0=fire、1=smoke。
- `run_train.py`：按作者训练入口设置的本机运行脚本；预计权重与日志写入 `runs/author_config_200ep/`。

## 数据核验

公开包内总计 16,885 张图像和同名 YOLO 标签：train 14,244，valid 2,641。train 有 1,747 个空标签文件，valid 有 1,001 个空标签文件，代表负样本；按标签框计数：train fire 25,868 / smoke 12,985；valid fire 3,055 / smoke 1,409。压缩包已成功完整解压，解压过程未报 CRC 错误。另发现 train 中 13 个宽或高为 0 的退化框；原始数据保持不变，待检查 Ultralytics 数据加载时是否会过滤。

## 启动

在此目录运行：

```powershell
.\.venv\Scripts\python.exe run_train.py
```

脚本沿用作者公开 `train.py` 的 200 epochs、640 输入、batch 16、SGD、seed 0、随机初始化（按用户要求从头训练；与作者脚本的 pretrained=True 不同）和主要增强/损失参数。考虑本机 Windows 与 16 GB 系统内存，workers 从作者配置的 16 调为 8。若 8 GB 显存不能承载 batch 16，会记录显存错误后降低 batch 并明确记录有效 batch 的差异。

## 状态

数据已下载、解压和配对检查。CUDA PyTorch 环境仍在安装中；在模型构建、单批次前后向、正式训练和验证成功前，不填写任何“复现指标”。
