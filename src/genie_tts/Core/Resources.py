import os


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

# ---- Run directory checks ----
ensure_exists(HUBERT_MODEL_DIR, "HUBERT_MODEL_DIR")
ensure_exists(SV_MODEL, "SV_MODEL")
# ensure_exists(ROBERTA_MODEL_DIR, "ROBERTA_MODEL_DIR")
