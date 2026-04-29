---
title: ocrloop
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ocrloop

A Telegram bot that extracts text from images with high precision and
preserves the original layout (line breaks, indentation, inter-word
spacing). Works on mixed Russian + English text and handles albums of up
to 20 photos in a single message.

## Highlights

- **Tesseract LSTM** with `rus+eng` language packs for accurate mixed-language
  recognition (default, lightweight).
- **Optional EasyOCR backend** for noticeably better script handling on noisy
  Russian screenshots (`OCR_ENGINE=easyocr`). Heavier — pulls PyTorch and
  downloads ~80 MB of model weights on first run.
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
| `OCR_ENGINE`    | `tesseract`   | Recognition backend: `tesseract` or `easyocr`.      |
| `OCR_LANGS`     | `rus+eng`     | Language packs to load (Tesseract codes; auto-mapped to ISO-639-1 for EasyOCR). |
| `OCR_LAYOUT`    | `compact`     | `compact` (drop blanks, flatten indent, prefix options with `О `) or `preserve` (keep reflowed layout). |
| `ALBUM_LATENCY` | `1.2`         | Seconds to wait for an album to fully arrive.       |

### Switching to the EasyOCR backend

```bash
pip install easyocr   # also pulls torch — ~3 GB on disk
export OCR_ENGINE=easyocr
python -m ocrloop.bot
```

First request after startup is slow (model load + first inference, ~10 s on
CPU). Subsequent requests are typically 1–4 seconds per image. EasyOCR is
better than Tesseract at picking the correct script in mixed Russian +
English documents (Tesseract's most common failure mode is recognising a
Russian word as all-Latin look-alikes, e.g. `Что → Yto`).

## Project layout

```
ocrloop/
├── __init__.py
├── album.py             # Telegram album buffering middleware
├── bot.py               # aiogram entry point and message handlers
├── confusables.py       # Latin↔Cyrillic post-processing
├── easyocr_backend.py   # Optional EasyOCR recognition + layout reflow
└── ocr.py               # Pre-processing, engine dispatch, layout reflow, cleanup
```

## Deploying

The repo ships a `Dockerfile` that installs Tesseract + RU/EN language packs.
The bot uses long polling (no webhook endpoint), so the natural fit is a
**Worker / Background Worker** service. If your host's free tier only
covers **Web Service** plans (e.g. Koyeb's free plan), the bot also
spins up a tiny no-op `/health` HTTP server when the `PORT` environment
variable is set — that satisfies the platform's TCP-port requirement
without changing how the bot itself communicates with Telegram.

### HuggingFace Spaces (free, no credit card)

HuggingFace Spaces gives you a free 24/7 Docker container without requiring
a payment method. The repo already has the Spaces YAML frontmatter at the
top of this README (`sdk: docker`, `app_port: 7860`) so HF picks it up
automatically.

1. Sign up at https://huggingface.co (no card needed).
2. **New** → **Space** → name it `ocrloop` → SDK: **Docker** → CPU basic
   (free) → Visibility: Private (so logs/secrets don't leak).
3. Add `BOT_TOKEN` as a **Repository secret** (Settings → Variables and
   secrets → New secret).
4. Push this repo to the Space's git remote:
   ```bash
   git clone https://github.com/MuhametaliSagiden/ocrloop.git ocrloop-hf
   cd ocrloop-hf
   git remote add hf https://huggingface.co/spaces/<your-user>/ocrloop
   git push hf main
   ```
   (You'll be prompted for your HF username and an access token created at
   https://huggingface.co/settings/tokens — give it `Write` permission.)
5. The Space builds (~5 min on first run). Once status is **Running**,
   send a photo to the bot in Telegram.

### Koyeb (free, no credit card)

1. Sign in at https://koyeb.com (GitHub OAuth).
2. **Create App** → **GitHub** → pick `MuhametaliSagiden/ocrloop`, branch
   `main`.
3. **Builder**: Dockerfile (auto-detected).
4. **Service type**: `Worker` if your plan allows it (no port needed).
   Otherwise `Web Service` — the bot will detect the `PORT` env var Koyeb
   injects and start the health endpoint automatically.
5. **Instance**: `Free` / `eco-nano`.
6. **Environment variables**: add `BOT_TOKEN` (required). All other vars
   (`OCR_ENGINE`, `OCR_LAYOUT`, `OCR_LANGS`, `ALBUM_LATENCY`) have sane
   defaults.
7. **Deploy**. First build ≈ 4 minutes (pulls Python + tesseract +
   language packs).

### Render (paid — Background Workers are no longer free)

Same Dockerfile works. New + → Background Worker → connect repo → Docker
runtime → set `BOT_TOKEN`. Render dropped the free tier for Background
Workers in 2024, so this requires a paid plan.

## How layout preservation works

`pytesseract.image_to_string` collapses consecutive whitespace, which loses
indentation. Instead we call `image_to_data` (TSV) to get every recognised
word together with its bounding box, then:

1. Compute the document's median character width.
2. Convert each word's left-edge pixel offset into a column number.
3. Re-emit each word at that column, padding with spaces.

This restores tabs / indentation that the original screenshot had.
