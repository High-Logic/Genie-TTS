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

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Lightweight Inference Engine

**Experience near-instantaneous speech synthesis on your CPU**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates TTS inference, ONNX model conversion, API
server, and other core features, aiming to provide ultimate performance and convenience.

* **✅ Supported Model Version:** GPT-SoVITS V2, V2ProPlus
* **✅ Supported Language:** Japanese, English, Chinese
* **✅ Supported Python Version:** >= 3.9

---

## 🎬 Demo Video

- **[➡️ Watch the demo video (Chinese)](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 Performance Advantages

GENIE optimizes the original model for outstanding CPU performance.

| Feature                     |  🔮 GENIE   | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to GENIE   |
| **Model Size**              | **\~230MB** |    Similar to GENIE    |       \~750MB       |

> 📝 **Latency Test Info:** All latency data is based on a test set of 100 Japanese sentences (\~20 characters each),
> averaged. Tested on CPU i7-13620H.

---

## 🏁 QuickStart

> **⚠️ Important:** It is recommended to run GENIE in **Administrator mode** to avoid potential performance degradation.

### 📦 Installation

Install via pip:

```bash
pip install genie-tts
```

## 📥 Pretrained Models

When running GENIE for the first time, it requires downloading resource files (**~391MB**). You can follow the library's
prompts to download them automatically.

> Alternatively, you can manually download the files
> from [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData)
> and place them in a local folder. Then set the `GENIE_DATA_DIR` environment variable **before** importing the library:

```python
import os

# Set the path to your manually downloaded resource files
# Note: Do this BEFORE importing genie_tts
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# The library will now load resources from the specified directory
```

### ⚡️ Quick Tryout

No GPT-SoVITS model yet? No problem!
GENIE includes several predefined speaker characters you can use immediately —
for example:

* **Mika (聖園ミカ)** — *Blue Archive* (Japanese)
* **ThirtySeven (37)** — *Reverse: 1999* (English)
* **Feibi (菲比)** — *Wuthering Waves* (Chinese)

You can browse all available characters here:
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](
https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

Try it out with the example below:

```python
import genie_tts as genie
import time

# Automatically downloads required files on first run
genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,  # Play the generated audio directly
)

genie.wait_for_playback_done()  # Ensure audio playback completes
```

### 🎤 TTS Best Practices

A simple TTS inference example:

```python
import genie_tts as genie

# Step 1: Load character voice model
genie.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Folder containing ONNX model
    language='<LANGUAGE_CODE>',  # Replace with language code, e.g., 'en', 'zh', 'jp'
)

# Step 2: Set reference audio (for emotion and intonation cloning)
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Must match loaded character name
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Path to reference audio
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Corresponding text
)

# Step 3: Run TTS inference and generate audio
genie.tts(
    character_name='<CHARACTER_NAME>',  # Must match loaded character
    text="<TEXT_TO_SYNTHESIZE>",  # Text to synthesize
    play=True,  # Play audio directly
    save_path="<OUTPUT_AUDIO_PATH>",  # Output audio file path
)

genie.wait_for_playback_done()  # Ensure audio playback completes

print("🎉 Audio generation complete!")
```

---

## 🔧 Model Conversion

To convert original GPT-SoVITS models for GENIE, ensure `torch` is installed:

```bash
pip install torch
```

Use the built-in conversion tool:

> **Tip:** `convert_to_onnx` currently supports V2 and V2ProPlus models.

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # Replace with your .pth file
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # Replace with your .ckpt file
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # Directory to save ONNX model
)
```

---

## 🌐 Launch FastAPI Server

GENIE includes a lightweight FastAPI server:

```python
import genie_tts as genie

# Start server
genie.start_server(
    host="0.0.0.0",  # Host address
    port=8000,  # Port
    workers=1  # Number of workers
)
```

> For request formats and API details, see our [API Server Tutorial](./Tutorial/English/API%20Server%20Tutorial.py).


---

## 📝 Roadmap

* [x] **🌐 Language Expansion**

    * [x] Add support for **Chinese** and **English**.

* [x] **🚀 Model Compatibility**

    * [x] Support for `V2Proplus`.
    * [ ] Support for `V3`, `V4`, and more.

* [x] **📦 Easy Deployment**

    * [ ] Release **Official Docker images**.
    * [x] Provide out-of-the-box **Windows bundles**.

---
