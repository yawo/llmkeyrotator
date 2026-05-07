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

- `POST /v1/*` — OpenAI-compatible proxy (e.g. `/v1/chat/completions`). Strips `/v1` prefix, forwards to provider `base_url + rest_of_path`. Always uses model from CSV.
- `POST /anthropic/*` — Anthropic-compatible proxy (e.g. `/anthropic/v1/messages`). Converts Anthropic ↔ OpenAI format, strips `/anthropic` prefix, forwards to provider. Always uses model from CSV.
- `GET /health` — Status, provider count, current provider
- `GET /stats` — Request/error/rotation counts, per-provider failure stats
- `/*` — Catch-all proxy to current provider (strips `/v1` or `/anthropic` prefix if present, always uses model from CSV)

## Build

```bash
# Native debug build (x86_64 Linux)
cargo build

# Native release build (x86_64 Linux)
cargo build --release
```

## Static Binaries

### Android ARM64 (via NDK)

```bash
# Prerequisites: Have Android NDK installed (tested with r29)
# Configure .cargo/config.toml with NDK toolchain paths

# Add target and build
rustup target add aarch64-linux-android
cargo build --release --target aarch64-linux-android
```

Output: `target/aarch64-linux-android/release/llmkeyrotator`

### Ubuntu x64 (musl)

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
