<div align="center">
<pre>
██████╗  ███████╗███╗   ██╗██╗███████╗
██╔════╝ ██╔════╝████╗  ██║██║██╔════╝
██║  ███╗█████╗  ██╔██╗ ██║██║█████╗  
██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝  
╚██████╔╝███████╗██║ ╚████║██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝
</pre>
</div>

<div align="center">

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**在 CPU 上体验近乎即时的语音合成**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** 是一个基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 构建的轻量级推理引擎。它集成了 TTS
推理、ONNX 模型转换、API 服务端以及其他核心功能，旨在提供极致的性能和便利性。

* **✅ 支持的模型版本：** GPT-SoVITS V2, V2ProPlus
* **✅ 支持的语言：** 日语、英语、中文、韩语
* **✅ 支持的 Python 版本：** >= 3.9

---

## 🎬 演示视频

- **[➡️ 观看演示视频（中文）](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 性能优势

GENIE 针对原始模型进行了优化，以实现出色的 CPU 性能。

| 特性         |  🔮 GENIE   | 官方 PyTorch 模型 | 官方 ONNX 模型 |
|:-----------|:-----------:|:-------------:|:----------:|
| **首次推理延迟** |  **1.13s**  |     1.35s     |   3.57s    |
| **运行时大小**  | **\~200MB** |    \~数 GB     | 与 GENIE 相似 |
| **模型大小**   | **\~230MB** |  与 GENIE 相似   |  \~750MB   |

> 📝 **延迟测试说明：** 所有延迟数据均基于 100 个日语句子（每句约 20 个字符）的测试集取平均值。测试环境为 CPU i7-13620H。

---

## 🏁 快速开始

> **⚠️ 重要提示：** 建议在 **管理员模式** 下运行 GENIE，以避免潜在的性能下降。

### 📦 安装

通过 pip 安装：

```bash
pip install genie-tts
```

## 📥 预训练模型

首次运行 GENIE 时，需要下载资源文件（**~391MB**）。您可以按照库的提示自动下载。

> 或者，您可以从 [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData) 手动下载文件并将其放置在本地文件夹中。然后在导入库
**之前** 设置 `GENIE_DATA_DIR` 环境变量：

```python
import os

# 设置手动下载的资源文件路径
# 注意：请在导入 genie_tts 之前执行此操作
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# 库现在将从指定目录加载资源
```

如果你想启用**仅用于中文推理**、用于改善中文韵律的可选 Chinese RoBERTa 文本特征，也可以这样下载：

```python
import genie_tts as genie

# 只下载可选的 Chinese RoBERTa 资源
genie.download_roberta_data()

# 或者直接走内置的完整资源下载流程，
# 该流程现在也会顺带下载可选的 Chinese RoBERTa 资源
genie.download_genie_data()
```

这些 RoBERTa 特征仅用于**中文**路径，以改善中文韵律；
它们**不应该用于**日语 / 英语 / 韩语推理。

### ⚡️ 快速试用

还没有 GPT-SoVITS 模型？没问题！
GENIE 包含几个预定义的说话人角色，您可以立即使用 —— 例如：

* **Mika (聖園ミカ)** — *蔚蓝档案 (Blue Archive)* (日语)
* **ThirtySeven (37)** — *重返未来：1999 (Reverse: 1999)* (英语)
* **Feibi (菲比)** — *鸣潮 (Wuthering Waves)* (中文)

您可以在此处浏览所有可用角色：
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](
https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

使用以下示例进行尝试：

```python
import genie_tts as genie
import time

# 首次运行时自动下载所需文件
genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,  # 直接播放生成的音频
)

genie.wait_for_playback_done()  # 确保音频播放完成
```

### 🎤 TTS 最佳实践

一个简单的 TTS 推理示例：

```python
import genie_tts as genie

# 第一步：加载角色语音模型
genie.load_character(
    character_name='<CHARACTER_NAME>',  # 替换为您的角色名称
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # 包含 ONNX 模型的文件夹
    language='<LANGUAGE_CODE>',  # 替换为语言代码，例如 'en', 'zh', 'jp', 'kr'
)

# 第二步：设置参考音频（用于情感和语调克隆）
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # 必须与加载的角色名称匹配
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # 参考音频的路径
    audio_text="<REFERENCE_AUDIO_TEXT>",  # 对应的文本
)

# 第三步：运行 TTS 推理并生成音频
genie.tts(
    character_name='<CHARACTER_NAME>',  # 必须与加载的角色匹配
    text="<TEXT_TO_SYNTHESIZE>",  # 要合成的文本
    play=True,  # 直接播放音频
    save_path="<OUTPUT_AUDIO_PATH>",  # 输出音频文件路径
)

genie.wait_for_playback_done()  # 确保音频播放完成

print("🎉 Audio generation complete!")
```

---

## 🔧 模型转换

要将原始 GPT-SoVITS 模型转换为 GENIE 格式，请确保已安装 `torch`：

```bash
pip install torch
```

使用内置的转换工具：

> **提示：** `convert_to_onnx` 目前支持 V2 和 V2ProPlus 模型。

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # 替换为您的 .pth 文件
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # 替换为您的 .ckpt 文件
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # 保存 ONNX 模型的目录
)
```

---

## 🌐 启动 FastAPI 服务

GENIE 包含一个轻量级的 FastAPI 服务器：

```python
import genie_tts as genie

# 启动服务
genie.start_server(
    host="0.0.0.0",  # 主机地址
    port=8000,  # 端口
    workers=1  # 工作进程数
)
```

> 关于请求格式和 API 详情，请参阅我们的 [API 服务教程](./Tutorial/English/API%20Server%20Tutorial.py)。


---

## 📝 路线图

* [x] **🌐 语言扩展**

    * [x] 添加对 **中文** 和 **英文** 的支持。

* [x] **🚀 模型兼容性**

    * [x] 支持 `V2ProPlus`。
    * [ ] 支持 `V3`、`V4` 等更多版本。

* [x] **📦 简易部署**

    * [ ] 发布 **官方 Docker 镜像**。
    * [x] 提供开箱即用的 **Windows 整合包**。

---
