# Norsk AI-detektor

Browser-app som klassifiserer norsk tekst som menneskeskrevet eller AI-generert.
All inferens kjører lokalt i nettleseren — teksten din forlater aldri maskinen.

🌐 **Live:** https://pegesund.github.io/norsk-ai-detektor

## Stack

- **Rust + Yew** (kompilert til WASM) for GUI
- **transformers.js** (JS) for ONNX-inferens og tokenisering
- **Pico.css** for styling
- Modell: [pegesund/norwegian-ai-detector](https://huggingface.co/pegesund/norwegian-ai-detector) (NorBERT-4-small fine-tunet)

## Lokal utvikling

Krever Rust + `trunk`:

```bash
cargo install --locked trunk
rustup target add wasm32-unknown-unknown
trunk serve
# Åpne http://127.0.0.1:8080
```

Bygg for produksjon:

```bash
trunk build --release
# Output i dist/
```

## Deploy

Automatisk via GitHub Actions ved push til `main` — bygger og deployer til GitHub Pages.

## Lisens

MIT
