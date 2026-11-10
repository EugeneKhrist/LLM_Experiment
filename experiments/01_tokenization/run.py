"""Опыт 1: одна и та же фраза на русском и английском через токенайзер.

Запуск: python experiments/01_tokenization/run.py
Результат: results.json и results.md рядом со скриптом.
"""
import json
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent

PHRASES = {
    "en": "Artificial intelligence is changing the way people work and learn.",
    "ru": "Искусственный интеллект меняет то, как люди работают и учатся.",
}

# Два токенайзера OpenAI: старый (GPT-3.5/4) и новый (GPT-4o)
ENCODINGS = ["cl100k_base", "o200k_base"]


def tokenize(enc, text):
    ids = enc.encode(text)
    # decode_single_token_bytes -> байты; для частей UTF-8 символа используем replace
    pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
    return ids, pieces


def main():
    results = []
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        for lang, text in PHRASES.items():
            ids, pieces = tokenize(enc, text)
            results.append(
                {
                    "encoding": enc_name,
                    "lang": lang,
                    "text": text,
                    "chars": len(text),
                    "bytes_utf8": len(text.encode("utf-8")),
                    "words": len(text.split()),
                    "tokens": len(ids),
                    "chars_per_token": round(len(text) / len(ids), 2),
                    "token_ids": ids,
                    "pieces": pieces,
                }
            )

    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# Опыт 1. Токенизация одной фразы на русском и английском\n"]
    lines.append(f"Библиотека: tiktoken {tiktoken.__version__}\n")
    lines.append("| Токенайзер | Язык | Символов | Слов | Токенов | Символов/токен |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['encoding']} | {r['lang']} | {r['chars']} | {r['words']} | "
            f"{r['tokens']} | {r['chars_per_token']} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['encoding']} / {r['lang']}\n")
        lines.append(f"Текст: {r['text']}\n")
        lines.append("Токены: " + " ".join(f"`[{p}]`" for p in r["pieces"]) + "\n")
        lines.append("ID: " + ", ".join(map(str, r["token_ids"])) + "\n")
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")

    for r in results:
        print(f"{r['encoding']:12} {r['lang']}  tokens={r['tokens']:3}  chars/token={r['chars_per_token']}")
        print("   ", " | ".join(r["pieces"]))


if __name__ == "__main__":
    main()
