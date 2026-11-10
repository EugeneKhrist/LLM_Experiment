"""Опыт 3: сравнение токенайзеров (OpenAI, DeepSeek-V3, Mistral Nemo) на разных текстах.

Тексты: фраза из опыта 1 (EN/RU/RU-латиница/KK) + два произвольных теста
(код/JSON и эмодзи/числа/символы).

Anthropic (Claude) здесь НЕ измеряется: у Claude нет локального токенайзера,
токены считает только API (messages/count_tokens), а ключа на машине нет.

Запуск:  python experiments/03_tokenizers/run.py
Папка с токенайзерами HF (необязательно): переменная HF_LOCAL (по умолчанию .cache/hf, в .gitignore).
"""
import json
import os
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).parent
HF_LOCAL = Path(os.environ.get("HF_LOCAL", HERE.parent.parent / ".cache" / "hf"))

TEXTS = {
    "en": "Artificial intelligence is changing the way people work and learn.",
    "ru": "Искусственный интеллект меняет то, как люди работают и учатся.",
    "ru_translit": "Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut i uchatsya.",
    "kk": "Жасанды интеллект адамдардың жұмыс істеу және оқу тәсілін өзгертіп жатыр.",
    # произвольные тесты
    "code": (
        "def fib(n: int) -> int:\n"
        "    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n"
        '{"user": {"id": 42, "tags": ["llm", "tokens"], "active": true}}'
    ),
    "emoji_numbers": "Цена: 1 234 567,89 ₽ (≈ $13 456.78) 🚀🔥 π=3.14159265358979 日本語 中文 العربية",
}

HF_MODELS = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
}


def tiktoken_tokenizers():
    out = {}
    for name in ["cl100k_base", "o200k_base", "o200k_harmony"]:
        try:
            enc = tiktoken.get_encoding(name)
        except Exception as e:  # o200k_harmony есть не во всех версиях
            print(f"skip {name}: {type(e).__name__}")
            continue

        def encode(text, enc=enc):
            ids = enc.encode(text, disallowed_special=())
            pieces = []
            for i in ids:
                b = enc.decode_single_token_bytes(i)
                try:
                    pieces.append(b.decode("utf-8"))
                except UnicodeDecodeError:
                    pieces.append("".join(f"<0x{x:02X}>" for x in b))
            return ids, pieces

        out[name] = encode
    return out


def hf_tokenizers():
    out = {}
    for name, repo in HF_MODELS.items():
        path = hf_hub_download(repo, "tokenizer.json", local_dir=HF_LOCAL / repo.replace("/", "__"))
        tok = Tokenizer.from_file(path)

        def encode(text, tok=tok):
            ids = tok.encode(text, add_special_tokens=False).ids
            # декодируем каждый id отдельно; обрывки байтов UTF-8 дают U+FFFD
            pieces = [tok.decode([i], skip_special_tokens=False).replace("�", "<byte>") for i in ids]
            return ids, pieces

        out[name] = encode
    return out


def main():
    tokenizers = {**tiktoken_tokenizers(), **hf_tokenizers()}
    results = []
    for tname, encode in tokenizers.items():
        for key, text in TEXTS.items():
            ids, pieces = encode(text)
            results.append(
                {
                    "tokenizer": tname,
                    "text_key": key,
                    "chars": len(text),
                    "bytes_utf8": len(text.encode("utf-8")),
                    "tokens": len(ids),
                    "chars_per_token": round(len(text) / len(ids), 2),
                    "pieces": pieces,
                }
            )
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list(tokenizers)
    by = {(r["tokenizer"], r["text_key"]): r for r in results}
    md = ["# Опыт 3. Сравнение токенайзеров\n"]
    md.append("Токенов на текст (меньше — экономнее). Anthropic/Claude не измерялся (нужен API-ключ).\n")
    md.append("| Текст | Символов | " + " | ".join(names) + " |")
    md.append("|---|---|" + "---|" * len(names))
    for key, text in TEXTS.items():
        row = [str(by[(names[0], key)]["chars"])] + [str(by[(n, key)]["tokens"]) for n in names]
        md.append(f"| {key} | " + " | ".join(row) + " |")
    md.append("")
    md.append("Символов на токен (больше — эффективнее):\n")
    md.append("| Текст | " + " | ".join(names) + " |")
    md.append("|---|" + "---|" * len(names))
    for key in TEXTS:
        md.append(f"| {key} | " + " | ".join(str(by[(n, key)]["chars_per_token"]) for n in names) + " |")
    md.append("")
    for key in ("code", "emoji_numbers"):
        md.append(f"## Разбивка: {key}\n")
        md.append("```\n" + TEXTS[key] + "\n```\n")
        for n in names:
            md.append(f"**{n}** ({by[(n, key)]['tokens']}): " + " ".join(f"`{p!r}`" for p in by[(n, key)]["pieces"]) + "\n")
    md.append("""## Выводы

- **Английский** почти не зависит от токенайзера: 11–12 токенов у всех.
- **o200k_harmony** (gpt-oss) на этих текстах совпал с o200k_base до токена: обычные токены те же, различия, видимо, только в служебных (не проверялось).
- **Русский**: o200k лучший (16), DeepSeek-V3 (20) и Mistral Nemo (19) в середине, cl100k худший (30).
- **Казахский** сильнее всего зависит от токенайзера: o200k — 19, Nemo — 30, DeepSeek — 40, cl100k — 59. Разброс в 3 раза на одном тексте.
- **Русский латиницей** везде ~25–27 токенов: ни один словарь не «знает» транслит как язык.
- **Код и JSON** — все ~55–56 токенов, токенайзеры здесь почти неотличимы.
- **Эмодзи, числа, CJK, арабский**: o200k лучший (36), Mistral Nemo худший (62, хуже даже старого cl100k — 50).
- **Anthropic/Claude не измерялся**: локального токенайзера нет, нужен `messages/count_tokens` и API-ключ. Сравнивать число токенов Claude с этими цифрами напрямую нельзя.
""")
    (HERE / "results.md").write_text("\n".join(md), encoding="utf-8")

    print("| text | chars | " + " | ".join(names))
    for key in TEXTS:
        print(f"{key:14} {by[(names[0], key)]['chars']:4} ", *[f"{by[(n, key)]['tokens']:4}" for n in names])


if __name__ == "__main__":
    main()
