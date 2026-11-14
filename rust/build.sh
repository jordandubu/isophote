#!/bin/sh
# build.sh — compile the wasm core and copy it into app/
set -e
cd "$(dirname "$0")"
cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/isophote_core.wasm ../app/isophote_core.wasm
echo "app/isophote_core.wasm $(du -h ../app/isophote_core.wasm | cut -f1)"
