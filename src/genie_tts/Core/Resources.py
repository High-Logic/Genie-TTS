import os
import sys
from huggingface_hub import snapshot_download


GENIE_DATA_REPO_ID = "High-Logic/Genie"
ROBERTA_REPO_ID = "litagin/chinese-roberta-wwm-ext-large-onnx"
ROBERTA_DIRNAME = "roberta-wwm-ext-large-onnx"

def _resolve_roberta_download(model_variant: str) -> tuple[str, list[str]]:
    variant = model_variant.lower().strip()
    if variant not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported roberta model_variant: {model_variant!r}")

    model_file = "model.onnx" if variant == "fp32" else "model_fp16.onnx"
    return model_file, [model_file, "tokenizer.json"]


def download_roberta_data(model_variant: str = "fp32") -> str:
    model_file, allow_patterns = _resolve_roberta_download(model_variant)
    target_dir = os.path.join(GENIE_DATA_DIR, ROBERTA_DIRNAME)

    print(
        f"Downloading Chinese RoBERTa ({model_variant}) to {target_dir}. "
        "This may take a while."
    )
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=ROBERTA_REPO_ID,
        repo_type="model",
        allow_patterns=allow_patterns,
        local_dir=target_dir,
        local_dir_use_symlinks=True,
    )
    print(f"Chinese RoBERTa downloaded successfully: {os.path.join(target_dir, model_file)}")
    return target_dir


def download_genie_data(include_roberta: bool = False, roberta_model_variant: str = "fp32") -> None:
    print("Starting download Genie-TTS resources. This may take a few moments.")
    snapshot_download(
        repo_id=GENIE_DATA_REPO_ID,
        repo_type="model",
        allow_patterns="GenieData/*",
        local_dir=".",
        local_dir_use_symlinks=True,
    )
    print("Genie-TTS resources downloaded successfully.")
    if include_roberta:
        download_roberta_data(model_variant=roberta_model_variant)


def ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required directory or file '{name}' was not found at: {path}\n"
            f"Please download the pretrained models and place them under './GenieData', "
            f"or set the environment variable GENIE_DATA_DIR to the correct directory."
        )


"""
文件结构与项目 Midori 同步。
"""

GENIE_DATA_DIR: str = os.getenv(
    "GENIE_DATA_DIR",
    "./GenieData"
)

"""
Japanese_G2P_DIR: str = os.getenv(
    "Japanese_G2P_DIR",
    f"{GENIE_DATA_DIR}/G2P/JapaneseG2P"
)
"""

English_G2P_DIR: str = os.getenv(
    "English_G2P_DIR",
    f"{GENIE_DATA_DIR}/G2P/EnglishG2P"
)

Chinese_G2P_DIR: str = os.getenv(
    "Chinese_G2P_DIR",
    f"{GENIE_DATA_DIR}/G2P/ChineseG2P"
)

HUBERT_MODEL_DIR: str = os.getenv(
    "HUBERT_MODEL_DIR",
    f"{GENIE_DATA_DIR}/chinese-hubert-base"
)

SV_MODEL: str = os.getenv(
    "SV_MODEL",
    f"{GENIE_DATA_DIR}/speaker_encoder.onnx"
)

ROBERTA_MODEL_DIR: str = os.getenv(
    "ROBERTA_MODEL_DIR",
    f"{GENIE_DATA_DIR}/RoBERTa"
)

if not os.path.exists(GENIE_DATA_DIR):
    print("GenieData folder not found.")
    if sys.stdin.isatty():
        choice = input("Would you like to download it automatically from HuggingFace? (y/N): ").strip().lower()
        if choice == "y":
            download_genie_data()
    else:
        print(
            "Non-interactive mode: skipping download prompt. "
            "Set GENIE_DATA_DIR env var or run interactively to configure."
        )

# ---- Run directory checks (skipped when GENIE_SKIP_RESOURCE_CHECK is set) ----
if not os.getenv("GENIE_SKIP_RESOURCE_CHECK"):
    ensure_exists(HUBERT_MODEL_DIR, "HUBERT_MODEL_DIR")
    ensure_exists(SV_MODEL, "SV_MODEL")
