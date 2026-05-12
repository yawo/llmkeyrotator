#!/bin/bash
set -euo pipefail

# Build static musl binaries for Ubuntu x64 and Android ARM64
#
# Prerequisites:
#   rustup target add x86_64-unknown-linux-musl aarch64-linux-android
#   sudo apt install musl-tools   # for x86_64 musl target

NDK=/home/yawo/android-sdk/ndk/29.0.14206865

echo "=== Building aarch64-linux-android (Android ARM64) ==="
rustup target add aarch64-linux-android 2>/dev/null || true
cargo build --release --target aarch64-linux-android
echo "=== Done: target/aarch64-linux-android/release/llmkeyrotator ==="
file target/aarch64-linux-android/release/llmkeyrotator

echo ""
echo "=== Building x86_64-unknown-linux-musl (Ubuntu x64) ==="
echo "NOTE: requires musl-gcc (sudo apt install musl-tools)"
rustup target add x86_64-unknown-linux-musl 2>/dev/null || true
if command -v musl-gcc &>/dev/null; then
    cargo build --release --target x86_64-unknown-linux-musl
    echo "=== Done: target/x86_64-unknown-linux-musl/release/llmkeyrotator ==="
    file target/x86_64-unknown-linux-musl/release/llmkeyrotator
else
    echo "SKIPPED: musl-gcc not found. Install with: sudo apt install musl-tools"
fi
cp target/aarch64-linux-android/release/llmkeyrotator llmkeyrotator-android
cp target/x86_64-unknown-linux-musl/release/llmkeyrotator llmkeyrotator-linux 
