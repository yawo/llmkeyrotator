# LLM Key Rotator

OpenAI-compatible proxy that rotates across providers on failure. Built with Rust + axum.

## Features

- **Per-failure rotation** through providers from CSV
- **Always uses CSV model** (ignores client's `model` field)
- **Streaming (SSE) + non-streaming** support
- **Structured logging** with `tracing`
- **`/health`** and **`/stats`** endpoints
- **Static musl binaries** for Android ARM64 (via NDK); Ubuntu x64 requires `musl-tools`

## CSV Format

```csv
name,base_url,model,api_key
gemini,https://generativelanguage.googleapis.com/v1beta/openai/,gemini-2.0-flash,AIza...
groq,https://api.groq.com/openai/v1,llama-3.3-70b,gsk_...
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_PATH` | `$HOME/code/freellmkeys.csv` | Path to provider CSV |
| `BASE_URL` | `http://0.0.0.0:3001/v1` | Bind address |
| `API_KEY` | (none) | Require `Bearer` auth header |
| `RUST_LOG` | `llmkeyrotator=info` | Log level |

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible, rotated across providers
- `GET /health` — Status, provider count, current provider
- `GET /stats` — Request/error/rotation counts, per-provider failure stats
- `/*` — Catch-all proxy to current provider

## Build

```bash
# Native debug build (x86_64 Linux)
cargo build

# Native release build (x86_64 Linux)
cargo build --release
```

## Static Binaries (musl)

### Android ARM64 (via NDK)

```bash
# Prerequisites: Have Android NDK installed (tested with r29)
export ANDROID_NDK_HOME=/home/yawo/android-sdk/ndk/29.0.14206865

# Add target and build
rustup target add aarch64-unknown-linux-musl
cargo build --release --target aarch64-unknown-linux-musl
```

Output: `target/aarch64-unknown-linux-musl/release/llmkeyrotator` (3.5MB statically-linked ELF)

### Ubuntu x64

```bash
# Prerequisites: Install musl-tools
# sudo apt install musl-tools

# Add target and build
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

Output: `target/x86_64-unknown-linux-musl/release/llmkeyrotator` (statically-linked ELF)

### Helper Script

```bash
./build.sh  # Attempts both targets, skips x86_64 if musl-gcc missing
```
