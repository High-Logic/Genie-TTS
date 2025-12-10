import os

Japanese_G2P_DIR: str = os.getenv(
    "English_G2P_DIR",
    "./GenieData/G2P/JapaneseG2P"
)

English_G2P_DIR: str = os.getenv(
    "English_G2P_DIR",
    "./GenieData/G2P/EnglishG2P"
)

Chinese_G2P_DIR: str = os.getenv(
    "Chinese_G2P_DIR",
    "./GenieData/G2P/ChineseG2P"
)
