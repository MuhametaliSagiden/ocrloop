# ocrloop

A Telegram bot that extracts text from images with high precision and
preserves the original layout (line breaks, indentation, inter-word
spacing). Works on mixed Russian + English text and handles albums of up
to 20 photos in a single message.

## Highlights

- **Tesseract LSTM** with `rus+eng` language packs for accurate mixed-language
  recognition.
- **Layout-preserving output** — words are reflowed from Tesseract's
  TSV bounding boxes onto a character grid so indentation, columns and tabs
  survive.
- **Album support** — sends one combined reply for albums of up to 20 photos,
  processed in the order they were sent.
- **Decorative-symbol stripping** — geometric bullets (•, ●, ◦, ▶, …) and
  bubble numerals (①, ❶, ⓵, …) are replaced with spaces so Tesseract doesn't
  hallucinate letters from them, while column alignment is preserved.
- **Clean output** — replies contain *only* the recognised text, no
  conversational filler.

## Quick start

### 1. System dependencies

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
```

### 2. Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and set BOT_TOKEN
```

### 4. Run

```bash
python -m ocrloop.bot
```

## Configuration

| Variable        | Default       | Description                                         |
| --------------- | ------------- | --------------------------------------------------- |
| `BOT_TOKEN`     | —             | Telegram bot token from @BotFather (required).      |
| `OCR_LANGS`     | `rus+eng`     | Tesseract language packs to load.                   |
| `ALBUM_LATENCY` | `1.2`         | Seconds to wait for an album to fully arrive.       |

## Project layout

```
ocrloop/
├── __init__.py
├── album.py    # Telegram album buffering middleware
├── bot.py      # aiogram entry point and message handlers
└── ocr.py      # Pre-processing, Tesseract call, layout reflow, cleanup
```

## How layout preservation works

`pytesseract.image_to_string` collapses consecutive whitespace, which loses
indentation. Instead we call `image_to_data` (TSV) to get every recognised
word together with its bounding box, then:

1. Compute the document's median character width.
2. Convert each word's left-edge pixel offset into a column number.
3. Re-emit each word at that column, padding with spaces.

This restores tabs / indentation that the original screenshot had.
