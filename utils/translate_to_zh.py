import translators as ts

def translate_to_zh(text) -> str:
    # 基础翻译
    result = ts.translate_text(text, translator='bing',from_language='en',to_language='zh')  # 可换为 'bing', 'baidu', 'deepl' 等
    return result