import numpy as np
from .Utils.Constants import BERT_FEATURE_DIM
from .ModelManager import model_manager


def get_phones_and_bert(prompt_text: str, language: str = 'japanese') -> tuple[np.ndarray, np.ndarray]:
    if language.lower() == 'english':
        from .G2P.English.EnglishG2P import english_to_phones
        phones = english_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)
    elif language.lower() == 'chinese':
        from .G2P.Chinese.ChineseG2P import chinese_to_phones
        text_clean, _, phones, word2ph = chinese_to_phones(prompt_text)
        if model_manager.load_roberta_model():
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            ort_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'repeats': np.array(word2ph, dtype=np.int64),
            }
            outputs = model_manager.roberta_model.run(None, ort_inputs)
            text_bert = outputs[0].astype(np.float32)
        else:
            text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)
    else:
        from .G2P.Japanese.JapaneseG2P import japanese_to_phones
        phones = japanese_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    phones_seq = np.array([phones], dtype=np.int64)
    return phones_seq, text_bert
