import re
import unicodedata
from calendar import month_name


# ------------------- 核心：自实现数字转单词 (替代 inflect) -------------------

def _number_to_words_custom(num_str):
    """一个不依赖inflect的、简化的数字到单词转换器。"""
    num_str = str(num_str).strip()
    if not num_str.isdigit(): return num_str

    num = int(num_str)
    if num == 0: return 'zero'

    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    thousands = ["", "thousand", "million", "billion", "trillion"]

    def convert_less_than_thousand(n):
        if n == 0: return ""
        if n < 10: return units[n]
        if n < 20: return teens[n - 10]
        if n < 100: return tens[n // 10] + (" " + units[n % 10] if n % 10 != 0 else "")
        return units[n // 100] + " hundred" + (" " + convert_less_than_thousand(n % 100) if n % 100 != 0 else "")

    words = []
    i = 0
    if num == 0: return "zero"
    while num > 0:
        if num % 1000 != 0:
            words.insert(0, convert_less_than_thousand(num % 1000) + " " + thousands[i])
        num //= 1000
        i += 1
    return " ".join(words).strip()


def _ordinal_custom(num_str):
    """一个不依赖inflect的、简化的序数词转换器。"""
    num = int(num_str)
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
    return _number_to_words_custom(str(num)) + suffix


# ------------------- 初始化和常量定义 (无 inflect) -------------------

_measurement_map = {
    "km/h": ["kilometer per hour", "kilometers per hour"], "mph": ["mile per hour", "miles per hour"],
    "°C": ["degree celsius", "degrees celsius"], "°F": ["degree fahrenheit", "degrees fahrenheit"],
    "tbsp": ["tablespoon", "tablespoons"], "tsp": ["teaspoon", "teaspoons"],
    "km": ["kilometer", "kilometers"], "kg": ["kilogram", "kilograms"], "min": ["minute", "minutes"],
    "ft": ["foot", "feet"], "cm": ["centimeter", "centimeters"], "m": ["meter", "meters"],
    "L": ["liter", "liters"], "h": ["hour", "hours"], "s": ["second", "seconds"],
}

_abbreviations = [
    (re.compile(r"\bMr\.(?=[\s,.]|\Z)", re.IGNORECASE), "Mister"),
    (re.compile(r"\bMrs\.(?=[\s,.]|\Z)", re.IGNORECASE), "Missus"),
    (re.compile(r"\bDr\.(?=[\s,.]|\Z)", re.IGNORECASE), "Doctor"),
    (re.compile(r"\bProf\.(?=[\s,.]|\Z)", re.IGNORECASE), "Professor"),
    (re.compile(r"\bSt\.(?=[\s,.]|\Z)", re.IGNORECASE), "Street"),
    (re.compile(r"\bCo\.(?=[\s,.]|\Z)", re.IGNORECASE), "Company"),
    (re.compile(r"\bLtd\.(?=[\s,.]|\Z)", re.IGNORECASE), "Limited"),
    (re.compile(r"\be\.g\.(?=[\s,.]|\Z)", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.(?=[\s,.]|\Z)", re.IGNORECASE), "that is"),
]

# ------------------- 正则表达式定义 (与原来保持一致) -------------------
_currency_suffix_re = re.compile(r"([£$€])([\d,.]*\d)\s*(million|billion|thousand)\b", re.IGNORECASE)
_phone_re = re.compile(r"(\+?\d{1,3}-)?\b(\d{3})-(?:(\d{3})-)?(\d{4})\b")
_roman_re = re.compile(r"\b(XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)\b", re.IGNORECASE)
_decade_re = re.compile(r"\b((?:1[89]|20)\d0)s\b")
_score_re = re.compile(r"\b(\d{1,2})-(\d{1,2})\b")
_dimension_re = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?\b")
_alphanumeric_re = re.compile(r"\b([a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\b")
_date_re = re.compile(r"\b(0?[1-9]|1[0-2])/([0-2]?\d|3[01])/(\d{2,4})\b")
_ordinal_number_re = re.compile(r"\b(\d+)\. ")
_comma_number_re = re.compile(r"(\d[\d,]+\d)")
_currency_re = re.compile(r"([£$€])(\d*\.?\d+)|(\d*\.?\d+)\s*([£$€])")
_time_re = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(\s*(?:a\.?m\.?|p\.?m\.?))?\b", re.IGNORECASE)
units = "|".join(re.escape(key) for key in sorted(_measurement_map.keys(), key=len, reverse=True))
_measurement_re = re.compile(rf"(?<!\w)(-?(?:\d+/\d+|\d+(?:\.\d+)?))\s*({units})\b")
_fraction_re = re.compile(r"\b(\d+)/(\d+)\b")
_decimal_number_re = re.compile(r"(\d+\.\d+)")
_ordinal_re = re.compile(r"\b\d+(st|nd|rd|th)\b")
_acronym_re = re.compile(r"\b[A-Z]{2,}\b")
_number_re = re.compile(r"(?<!\w)-?\d+(?!\w)")


# ------------------- 替换与扩展函数 (全部使用 _number_to_words_custom) -------------------
def _expand_currency_suffix(m):
    symbol, amount_str, suffix = m.groups()
    major_map = {"$": "dollars", "£": "pounds", "€": "euros"}
    amount_word = _number_to_words_custom(amount_str.replace(",", ""))
    currency_word = major_map.get(symbol, "")
    return f"{amount_word} {suffix} {currency_word}"


def _expand_phone_number(m):
    country, area, exch, line = m.groups()
    parts = []
    if country:
        country_words = []
        if country.startswith('+'): country_words.append('plus')
        digits_only = re.sub(r'\D', '', country)
        if digits_only: country_words.append(' '.join(_number_to_words_custom(d) for d in digits_only))
        parts.append(' '.join(country_words))
    parts.append(' '.join(_number_to_words_custom(c) for c in area))
    if exch: parts.append(' '.join(_number_to_words_custom(c) for c in exch))
    parts.append(' '.join(_number_to_words_custom(c) for c in line))
    return ", ".join(parts)


def _expand_roman(m):
    roman_map = {
        "ii": "two", "iii": "three", "iv": "four", "v": "five", "vi": "six", "vii": "seven",
        "viii": "eight", "ix": "nine", "x": "ten", "xi": "eleven", "xii": "twelve",
        "xiii": "thirteen", "xiv": "fourteen", "xv": "fifteen", "xvi": "sixteen",
        "xvii": "seventeen", "xviii": "eighteen", "xix": "nineteen"
    }
    return roman_map.get(m.group(1).lower(), m.group(1))


def _expand_decade(m):
    year_str = m.group(1)
    year_words = _expand_number_positive(year_str)
    if year_words.endswith('ty'): return f"{year_words[:-1]}ies"
    return f"{year_words}s"


def _expand_dimension(m):
    parts = [p for p in m.groups() if p is not None]
    expanded_parts = [_number_to_words_custom(p) for p in parts]
    return " by ".join(expanded_parts)


def _expand_score(m):
    return f"{_number_to_words_custom(m.group(1))} to {_number_to_words_custom(m.group(2))}"


def _expand_alphanumeric(m):
    text = m.group(0)
    parts = re.findall(r'[a-zA-Z]+|[0-9]+', text)
    expanded_parts = []
    for part in parts:
        if part.isalpha():
            expanded_parts.append(' '.join(list(part)))
        elif part.isdigit():
            expanded_parts.append(' '.join(_number_to_words_custom(c) for c in part))
    return ' '.join(expanded_parts)


def _convert_ordinal(m):
    return _ordinal_custom(m.group(1)) + ", "


def _remove_commas(m): return m.group(1).replace(",", "")


def _expand_time(m):
    h_str, m_str, s_str, am_pm = m.groups()
    h, m = int(h_str), int(m_str)
    h_word = _number_to_words_custom(h if h <= 12 or not am_pm else h - 12)
    if h == 0 and am_pm: h_word = "twelve"
    m_word = ""
    if m > 0: m_word = f" oh {_number_to_words_custom(m)}" if m < 10 else f" {_number_to_words_custom(m)}"
    result = f"{h_word}{m_word}".lstrip()
    if s_str: result += f" and {_number_to_words_custom(int(s_str))} seconds"
    if am_pm: result += ' pm' if 'p' in am_pm.lower() else ' am'
    return result


def _expand_measurement(m):
    num_str, unit = m.groups()
    is_neg = num_str.startswith('-')
    if is_neg: num_str = num_str[1:]
    if '/' in num_str:
        num_word = _expand_fraction(re.match(_fraction_re, num_str))
        is_plural = True
    else:
        num_word = _number_to_words_custom(num_str)
        is_plural = float(num_str) != 1
    unit_word = _measurement_map[unit][1] if is_plural else _measurement_map[unit][0]
    result = f"{num_word} {unit_word}"
    return f"minus {result}" if is_neg else result


def _expand_currency(m):
    symbol, amount_str = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    amount_str = (amount_str or "").replace(",", "")
    if amount_str.startswith('.'): amount_str = '0' + amount_str
    major_map = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}
    minor_map = {"$": ("cent", "cents"), "£": ("penny", "pence"), "€": ("cent", "cents")}
    major_singular, major_plural = major_map.get(symbol, ("", ""))
    parts = amount_str.split('.')
    major_val = int(parts[0]) if parts[0] else 0
    minor_val = int(parts[1].ljust(2, '0')) if len(parts) > 1 and parts[1] else 0
    result = []
    if major_val > 0:
        result.append(f"{_number_to_words_custom(major_val)} {major_singular if major_val == 1 else major_plural}")
    if minor_val > 0:
        minor_singular, minor_plural = minor_map.get(symbol, ("", ""))
        result.append(f"{_number_to_words_custom(minor_val)} {minor_singular if minor_val == 1 else minor_plural}")
    return " and ".join(result) or f"zero {major_plural}"


def _expand_decimal_number(m):
    num_str = m.group(1)
    parts = num_str.split('.')
    integer_part = _number_to_words_custom(parts[0])
    fractional_part = ' '.join(_number_to_words_custom(digit) for digit in parts[1])
    return f"{integer_part} point {fractional_part}"


def _expand_date(m):
    month, day, year = m.groups()
    month_word = month_name[int(month)]
    day_word = _ordinal_custom(day)
    year_num = int(year)
    if len(year) == 2: year_num += 2000 if year_num < 50 else 1900
    return f"{month_word} {day_word}, {_expand_number_positive(str(year_num))}"


def _expand_fraction(m):
    n, d = int(m.group(1)), int(m.group(2))
    if d == 0: return m.group(0)
    common_fractions = {(1, 2): "one half", (1, 4): "one quarter", (3, 4): "three quarters"}
    if (n, d) in common_fractions: return common_fractions[(n, d)]
    return f"{_number_to_words_custom(n)} over {_number_to_words_custom(d)}"


def _expand_ordinal_word(m):
    return _ordinal_custom(m.group(0)[:-2])


def _expand_number(m):
    num_str = m.group(0)
    if num_str.startswith('-'): return f"minus {_expand_number_positive(num_str[1:])}"
    return _expand_number_positive(num_str)


def _expand_number_positive(num_str):
    num = int(num_str)
    if 2000 <= num < 2010: return f"two thousand and {_number_to_words_custom(num % 100)}"
    if 1100 <= num < 2100 and num % 100 != 0:
        return f"{_number_to_words_custom(num // 100)} {_number_to_words_custom(num % 100)}"
    return _number_to_words_custom(num_str)


def _expand_acronym(m): return " ".join(m.group(0))


def normalize(text):
    text = "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")
    text = re.sub(r"@", " at ", text)
    for regex, replacement in _abbreviations: text = regex.sub(replacement, text)
    text = re.sub(_currency_suffix_re, _expand_currency_suffix, text)
    text = re.sub(_phone_re, _expand_phone_number, text)
    text = re.sub(_dimension_re, _expand_dimension, text)
    text = re.sub(_roman_re, _expand_roman, text)
    text = re.sub(_decade_re, _expand_decade, text)
    text = re.sub(_score_re, _expand_score, text)
    text = re.sub(_date_re, _expand_date, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_ordinal_number_re, _convert_ordinal, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_ordinal_re, _expand_ordinal_word, text)
    text = re.sub(_alphanumeric_re, _expand_alphanumeric, text)
    text = re.sub(_acronym_re, _expand_acronym, text)
    text = re.sub(_number_re, _expand_number, text)
    text = text.lower()
    text = re.sub(r"%", " percent", text)
    domain_re = re.compile(r'\b([a-z0-9-]+)\.([a-z]{2,})\b')
    while domain_re.search(text): text = domain_re.sub(r'\1 dot \2', text)
    text = re.sub(r"[^a-z0-9'.,?!:;-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
