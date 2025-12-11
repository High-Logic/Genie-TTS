"""
该文件仅供开发者使用。
请在 Pycharm 把 src 文件夹 Mark 为 Sources Root, 这样才能导入 genie_tts 并进行测试。

2025/12/11
测试项目:
- V2 转换
- V2 ProPlus 转换
- 中文推理
- 日文推理
- 英文推理
全部通过。
"""
# 将 Genie 链接到 Midori
import os

os.environ['GENIE_DATA_DIR'] = r"C:\Users\Haruka\Desktop\Midori\Data\common_resource\tts"

import genie_tts as genie
import shutil

# genie.set_log_severity_level(logging.ERROR)

TEST_CACHE_DIR = './GenieTemp'
os.makedirs(TEST_CACHE_DIR, exist_ok=True)
MIDORI_DIR = 'C:/Users/Haruka/Desktop/Midori'
V2ProPlus = {
    'Chinese': {
        'genie_model_dir': f'{MIDORI_DIR}/Data/character_resource/hutao/tts/tts_models',
        'gpt_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/hutao/Hutao-e50.ckpt",
        'vits_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/hutao/Hutao_e25_s2225.pth",
        'default_text': '棉花大哥哥和鬼蜀黍真是一对苦命鸳鸯啊。',
        'prompt_audio': f"{MIDORI_DIR}/Data/character_resource/hutao/tts/prompt_wav/vo_SGEQ002_15_hutao_06.wav",
        'prompt_text': '行了行了，不用那么谦虚。不知道的还以为你紧张了。尽管点评就行了！'
    },

    'Japanese': {
        'genie_model_dir': f'{MIDORI_DIR}/Data/character_resource/miyo/tts/tts_models',
        'gpt_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/miyo/Miyo-e50.ckpt",
        'vits_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/miyo/Miyo_e25_s425.pth",
        'default_text': '今日も素敵な一日をお過ごしください。',
        'prompt_audio': f"{MIDORI_DIR}/Data/character_resource/miyo/tts/prompt_wav/861850.wav",
        'prompt_text': '先生を見ながら、お話を書きたいです…理由がなきゃ…だめですか？'
    },

    'English': {
        'genie_model_dir': f'{MIDORI_DIR}/Data/character_resource/sonetto/tts/tts_models',
        'gpt_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/sonetto/sonetto-e30.ckpt",
        'vits_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2ProPlus)/sonetto/sonetto_e24_s240.pth",
        'default_text': 'Hello! Welcome to our language learning platform.',
        'prompt_audio': f"{MIDORI_DIR}/Data/character_resource/sonetto/tts/prompt_wav/162310.wav",
        'prompt_text': 'Are you a brave speaker?You look like one -- hmm, and sound like too.'
    }
}
V2 = {
    'Japanese': {
        'gpt_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2)/nene/nene.ckpt",
        'vits_path': f"{MIDORI_DIR}/Data/gpt_sovits_resource (private)/Model (V2)/nene/nene.pth",
        'default_text': '今日も素敵な一日をお過ごしください。',
        'prompt_audio': f"{MIDORI_DIR}/Data/character_resource/nene/tts/prompt_wav/nen001_010.wav",
        'prompt_text': '何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも'
    },
}


def test_convert(cfg: dict, lang: str = 'Chinese', character_name='Test'):
    print(f'开始测试 convert')

    if os.path.isdir(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)

    genie.convert_to_onnx(
        torch_ckpt_path=cfg[lang]['gpt_path'],
        torch_pth_path=cfg[lang]['vits_path'],
        output_dir=os.path.join(TEST_CACHE_DIR, 'TempModel'),
    )
    genie.load_character(
        character_name=character_name,
        onnx_model_dir=os.path.join(TEST_CACHE_DIR, 'TempModel'),
        language=lang,
    )
    genie.set_reference_audio(
        character_name=character_name,
        audio_path=cfg[lang]['prompt_audio'],
        audio_text=cfg[lang]['prompt_text'],
    )
    while True:
        text = input('输入文本(输入c停止):')
        if text.lower() == 'c':
            break
        genie.tts(
            character_name=character_name,
            text=text or cfg[lang]['default_text'],
            play=True,
        )
        genie.wait_for_playback_done()

    print(f'结束测试 v2pp convert')


def test_tts(cfg: dict, lang: str = 'Chinese', character_name: str = 'Test'):
    print(f'开始测试 {lang} TTS')

    genie.load_character(
        character_name=character_name,
        onnx_model_dir=cfg[lang]['genie_model_dir'],
        language=lang,
    )
    genie.set_reference_audio(
        character_name=character_name,
        audio_path=cfg[lang]['prompt_audio'],
        audio_text=cfg[lang]['prompt_text'],
    )
    genie.tts(
        character_name=character_name,
        text=cfg[lang]['default_text'],
        play=True,
    )
    genie.wait_for_playback_done()

    print(f'结束测试 {lang} TTS')
    genie.unload_character(character_name)


def test_tts_all():
    test_tts(V2ProPlus, lang='Chinese')
    test_tts(V2ProPlus, lang='English')
    test_tts(V2ProPlus, lang='Japanese')


if __name__ == '__main__':
    try:
        test_tts(V2ProPlus, lang='Japanese')
        # test_convert(V2ProPlus, lang='Japanese')
        # test_convert(V2, lang='Japanese')
    finally:
        print('删除临时文件夹')
        if os.path.isdir(TEST_CACHE_DIR):
            shutil.rmtree(TEST_CACHE_DIR)
