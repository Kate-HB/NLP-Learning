# NLP 学习环境配置

## 1. 当前环境

本项目使用 D 盘 Conda 的 `base` 环境。

| 项目 | 当前配置 |
|---|---|
| Conda 根目录 | `D:\Soft\Conda` |
| Python 解释器 | `D:\Soft\Conda\python.exe` |
| Python | `3.13.9` |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA Runtime | `12.8` |
| Transformers | `4.57.3` |
| Datasets | `5.0.0` |
| GPU | 可用 |
| JupyterLab | 已安装 |

Transformers 固定为 `4.57.3`，用于兼容 Hugging Face LLM Course 中的 `question-answering` 等 Pipeline 示例。

## 2. 激活环境

在 PowerShell 或 VS Code 终端中执行：

```powershell
conda activate base
```

检查当前 Python 路径：

```powershell
where.exe python
python -c "import sys; print(sys.executable)"
```

正确结果应包含：

```text
D:\Soft\Conda\python.exe
```

若 `conda` 命令不可用，可直接调用：

```powershell
D:\Soft\Conda\Scripts\conda.exe activate base
```

## 3. VS Code 配置

项目的 `.vscode/settings.json` 已绑定 D 盘 Conda：

```json
{
  "python-envs.defaultEnvManager": "ms-python.python:conda",
  "python-envs.defaultPackageManager": "ms-python.python:conda",
  "python.defaultInterpreterPath": "D:\\Soft\\Conda\\python.exe"
}
```

确认解释器：

1. 按 `Ctrl + Shift + P`。
2. 执行 `Python: Select Interpreter`。
3. 选择 `D:\Soft\Conda\python.exe`。

Notebook 还需要在右上角的 Kernel 菜单中选择同一个解释器。

## 4. PyTorch 与 CUDA 验证

在终端执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

当前预期结果：

```text
2.11.0+cu128
12.8
True
```

查看显卡名称：

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

说明：

- `+cu128` 表示 PyTorch 包含 CUDA 12.8 运行库。
- 不需要为每个 Conda 环境重新安装完整 CUDA Toolkit。
- 必须存在兼容的 NVIDIA 显卡驱动。
- `torch.cuda.is_available()` 为 `True` 才表示 PyTorch 能使用 GPU。

## 5. Hugging Face 依赖验证

查看版本：

```powershell
python -c "import transformers, datasets; print(transformers.__version__); print(datasets.__version__)"
```

检查依赖冲突：

```powershell
python -m pip check
```

当前预期结果：

```text
No broken requirements found.
```

不要直接执行无版本限制的升级：

```text
pip install --upgrade transformers
```

这可能升级到与课程示例不兼容的版本。本项目使用：

```powershell
python -m pip install "transformers==4.57.3"
```

安装或切换版本后，必须重启 Jupyter Kernel。

## 6. Jupyter 使用

### 浏览器中启动

进入项目目录：

```powershell
cd C:\Users\27729\Desktop\NLP-Learning
conda activate base
jupyter lab
```

终端会输出本地地址，浏览器通常自动打开。

停止服务：

```text
Ctrl + C
```

### VS Code 中使用

1. 打开 `.ipynb` 文件。
2. 点击右上角 Kernel。
3. 选择 `D:\Soft\Conda\python.exe`。
4. 按 `Shift + Enter` 运行单元格。

检查 Notebook 使用的环境：

```python
import sys
import torch
import transformers

print(sys.executable)
print(torch.__version__)
print(transformers.__version__)
```

## 7. Hugging Face 登录

在 VS Code 终端执行：

```powershell
hf auth login
```

根据终端提示打开浏览器、输入验证码并授权。

验证登录状态：

```powershell
hf auth whoami
```

登录后，Transformers 和 `huggingface_hub` 会自动读取本机凭证。不要把 Token 写入代码、Notebook、README 或 Git。

退出登录：

```powershell
hf auth logout
```

## 8. 模型缓存

首次使用模型时，Hugging Face 会下载模型文件到本地缓存。Windows 默认位置通常为：

```text
C:\Users\27729\.cache\huggingface\hub
```

之后再次加载相同模型时会复用缓存。

查看缓存信息：

```powershell
hf cache ls
```

模型可能占用数百 MB 到数十 GB。下载前应检查模型仓库的文件大小和格式。

## 9. Windows 符号链接警告

若出现：

```text
To support symlinks on Windows, you either need to activate Developer Mode...
```

这是缓存优化警告，不是运行错误。处理方式：

```text
Windows 设置 → 系统 → 开发者选项 → 开发人员模式
```

开启后重启 VS Code。不要长期以管理员身份运行 Python。

## 10. 最小验证程序

在终端运行：

```powershell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')('I love NLP'))"
```

预期输出类似：

```text
[{'label': 'POSITIVE', 'score': 0.9997}]
```

在 Notebook 中验证：

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

classifier("I love NLP")
```

## 11. 常见问题

### VS Code 终端显示 `(base)`，Notebook 却找不到包

终端环境和 Notebook Kernel 是两套选择。重新选择 Notebook 右上角 Kernel：

```text
D:\Soft\Conda\python.exe
```

### 安装包后 Notebook 仍显示旧版本

重启 Kernel，再重新运行：

```python
import transformers
print(transformers.__version__)
```

### `torch.cuda.is_available()` 返回 `False`

依次检查：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

确认安装的是带 `+cu128` 的 PyTorch，而不是 CPU 版。

### `Unknown task question-answering`

检查 Transformers 版本：

```powershell
python -c "import transformers; print(transformers.__version__)"
```

本项目应为：

```text
4.57.3
```

### `Should have a model_type key in its config.json`

该模型可能不是 Transformers 格式，例如 GGUF 或 Flair 模型。应先查看模型页的框架标签和使用说明。

### `No mask_token found`

检查 Tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
print(tokenizer.mask_token)
```

若结果为 `None`，模型不支持 `fill-mask`。

## 12. 每次学习前检查

```powershell
conda activate base
python -c "import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(transformers.__version__)"
```

应确认：

- Python 路径为 `D:\Soft\Conda\python.exe`
- PyTorch 版本包含 `+cu128`
- CUDA 可用为 `True`
- Transformers 为 `4.57.3`
