import re

# --- Constants for English Sentence Splitting ---

# A shorter minimum length for the first sentence to reduce TTS latency.
MIN_FIRST_SENTENCE_LENGTH = 10
# Minimum valid character length for a regular sentence.
MIN_SENTENCE_LENGTH = 15

# Terminators for splitting the FIRST sentence. Includes comma for a quicker split.
FIRST_SENTENCE_TERMINATORS = ",.!?;"
# Standard sentence terminators for the rest of the text.
SENTENCE_TERMINATORS = ".!?"
# Punctuation that can follow a terminator, like quotes or brackets.
CLOSING_PUNCTUATION = "\"'`”’)}\\]"

# Regex pattern to find what we consider a "valid" character for length calculation.
# For English, this is primarily letters and numbers.
VALID_CHAR_PATTERN_EN = re.compile(r'[a-zA-Z0-9]')


def get_valid_text_length_en(sentence: str) -> int:
    """
    Calculates the length of a string based on valid English characters (letters and numbers).
    """
    return len(VALID_CHAR_PATTERN_EN.findall(sentence))


def _merge_short_sentences_en(sentences_list: list[str]) -> list[str]:
    """
    Merges adjacent short sentences based on MIN_SENTENCE_LENGTH.
    This is crucial for TTS to avoid many small, disjointed audio clips.
    """
    if not sentences_list:
        return []

    merged_list = []
    temp_sentence = ""

    for sentence in sentences_list:
        if temp_sentence:
            temp_sentence += " " + sentence.lstrip()  # 加上空格，并移除被拼接句子的前导空格
        else:
            temp_sentence = sentence
        if get_valid_text_length_en(temp_sentence) >= MIN_SENTENCE_LENGTH:
            merged_list.append(temp_sentence)
            temp_sentence = ""

    # If there's a leftover short sentence, append it to the last one.
    if temp_sentence:
        if merged_list:
            merged_list[-1] += " " + temp_sentence.lstrip()
        else:
            merged_list.append(temp_sentence)

    return merged_list


def split_english_text(long_text: str) -> list[str]:
    """
    Splits a long English text into a list of sentences, optimized for TTS.
    """
    if not long_text or not long_text.strip():
        return []

    # --- Step 1: Handle the first sentence with relaxed rules ---
    first_sentence = None
    remaining_text = long_text.strip()

    # Pattern: one or more terminators, followed by optional closing punctuation and whitespace.
    first_split_pattern = re.compile(f'[{FIRST_SENTENCE_TERMINATORS}]+[{CLOSING_PUNCTUATION}]*\\s*')

    for match in first_split_pattern.finditer(remaining_text):
        end_pos = match.end()
        potential_first = remaining_text[:end_pos]

        if get_valid_text_length_en(potential_first) >= MIN_FIRST_SENTENCE_LENGTH:
            first_sentence = potential_first.strip()
            remaining_text = remaining_text[end_pos:].strip()
            break

    # --- Step 2: Process the rest of the text with standard rules ---
    other_sentences = []
    if remaining_text:
        # Standard end pattern: one or more terminators, followed by optional closing punctuation.
        end_pattern = re.compile(f'[{SENTENCE_TERMINATORS}]+[{CLOSING_PUNCTUATION}]*\\s*')

        raw_other = []
        last_end = 0

        for match in end_pattern.finditer(remaining_text):
            sentence = remaining_text[last_end:match.end()]
            raw_other.append(sentence.strip())
            last_end = match.end()

        # Add the final part of the text if it doesn't end with a terminator.
        if last_end < len(remaining_text):
            raw_other.append(remaining_text[last_end:].strip())

        # Clean up any potential empty strings.
        raw_other = [s for s in raw_other if s]

        if raw_other:
            # Merge sentences that are too short.
            other_sentences = _merge_short_sentences_en(raw_other)

    # --- Step 3: Combine and finalize the results ---
    final_result = []
    if first_sentence:
        final_result.append(first_sentence)

    if other_sentences:
        final_result.extend(other_sentences)
    # If no sentences were split but text remains (e.g., no punctuation).
    elif not first_sentence and remaining_text:
        return [remaining_text]

    return [s for s in final_result if s]


if __name__ == "__main__":
    print("--- Starting Comprehensive English Splitting Tests ---")

    test_cases = {
        # --- 1. First Sentence Special Splitting Logic ---
        "First Sentence: Split by comma, meets length": "Hello there, this is a test to see how the initial splitting works. The rest of the text follows.",
        "First Sentence: First comma too short, uses period": "Yes, but this is the full sentence. It should not split after 'Yes,'.",
        "First Sentence: Split by semicolon": "This is the first part; and this is the second part.",

        # --- 2. Punctuation and Quote Handling ---
        "Multiple Punctuation: Question and Exclamation": "What is this?! I can't believe it!",
        "Ellipsis: Ending with '...'": "He just stood there, staring...",
        "Quotes: Ending with a quote": 'She said, "I will be there." Then she left.',
        "Quotes: Quote at the very end": 'He asked, "Are you sure about this?"',
        "Parentheses: Sentence ending in parentheses": "This is a test (a simple one). This is another sentence.",

        # --- 3. Short Sentence Merging Logic ---
        "Merging: Multiple short sentences": "Yes. No. Maybe. I don't know. Let's think about it.",
        "Merging: Long sentence followed by short ones": "This is the first sentence. OK. Let's go.",
        "Merging: Not merging if length is sufficient": "I think. Therefore I am.",
        # "I think." is short, but the next is also short, they should merge. Let's try another.
        "Merging: Just long enough to not merge": "This sentence is just long enough. This one too.",

        # --- 4. Edge Cases ---
        "Edge Case: Empty String": "",
        "Edge Case: Whitespace only": "   \t\n  ",
        "Edge Case: No punctuation at all": "This is a single long sentence without any ending punctuation",
        "Edge Case: No punctuation at the end": "This is the first sentence. This is the second and final part",
        "Edge Case: Text starts with punctuation": "!!Look at this. It's starting with a bang.",
        "Edge Case: Single short sentence": "Wow!",
        "Edge Case: Text is only punctuation": ".?!,;...",

        # --- 5. Real-world examples ---
        "Complex Case 1: Abbreviations (known issue)": "Dr. Smith went to Washington, D.C. to see the new exhibit. It was amazing.",
        "Complex Case 2: List-like structure": "First, we need to gather the requirements. Second, we design the system. Third, we implement it.",
    }

    total_tests = len(test_cases)
    passed_count = 0  # In this context, "passed" means it ran without error.

    print(f"Total test cases: {total_tests}\n")

    for i, (description, text) in enumerate(test_cases.items()):
        print(f"--- [Test {i + 1}/{total_tests}] Scenario: {description} ---")
        print(f"Original: {text}")
        try:
            result = split_english_text(text)
            print(f"Split Result ({len(result)} sentences):")
            if not result and text.strip() != "" and not re.fullmatch(r'[\s\W]+', text):
                print("  >> [WARNING] Text provided but result is empty!")
            for j, s in enumerate(result):
                print(f"  {j + 1}: {s} (Valid Chars: {get_valid_text_length_en(s)})")
            passed_count += 1
        except Exception as e:
            print(f"  !!!!!! [ERROR] Test failed: {e} !!!!!!")
        print("--------------------------------------------------\n")

    print("--- Testing Complete ---")
    print(f"Result: {passed_count} / {total_tests} test cases processed.")
