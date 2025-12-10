"""
该文件仅供开发者使用。
请在 Pycharm 把 src 文件夹 Mark 为 Sources Root, 这样才能导入 genie_tts 并进行测试。
"""
# 链接到 Midori
import os

os.environ['GENIE_DATA_DIR'] = r"C:\Users\Haruka\Desktop\Midori\Data\common_resource\tts"

import genie_tts as genie

# 自行修改
MIDORI_DIR = 'C:/Users/Haruka/Desktop/Midori'
TTS_CONFIGS = {
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


def test_tts(lang: str = 'Chinese', character_name: str = 'Test'):
    print(f'开始测试 {lang} TTS')
    genie.load_character(
        character_name=character_name,
        onnx_model_dir=TTS_CONFIGS[lang]['genie_model_dir'],
        language=lang,
    )
    genie.set_reference_audio(
        character_name=character_name,
        audio_path=TTS_CONFIGS[lang]['prompt_audio'],
        audio_text=TTS_CONFIGS[lang]['prompt_text'],
    )
    genie.tts(
        character_name=character_name,
        text=TTS_CONFIGS[lang]['default_text'],
        play=True,
    )
    genie.wait_for_playback_done()
    print(f'{lang} TTS 测试结束')


if __name__ == '__main__':
    test_tts('Chinese')
