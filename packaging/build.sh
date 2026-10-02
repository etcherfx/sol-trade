#!/usr/bin/env bash
# Builds the release archives in dist/: one per OS, holding only what SolTrade needs to run
# (committed files at HEAD) plus that OS's launcher. Needs git, tar and zip.
# Usage: packaging/build.sh VERSION
set -euo pipefail

version="${1:?usage: packaging/build.sh VERSION}"
cd "$(dirname "$0")/.."
name="sol-trade-$version"
files=(main.py sol_trade strategies pyproject.toml uv.lock .python-version config.json.sample .env.sample README.md LICENSE)

rm -rf dist
for os in windows macos-linux; do
  mkdir -p "dist/stage/$os/$name"
  git archive HEAD -- "${files[@]}" | tar -x -C "dist/stage/$os/$name"
done
cp packaging/run.cmd packaging/run.ps1 "dist/stage/windows/$name/"
install -m 755 packaging/run.sh "dist/stage/macos-linux/$name/run.sh"

(cd dist/stage/windows && zip -qr "../../$name-windows.zip" "$name")
tar -czf "dist/$name-macos-linux.tar.gz" -C dist/stage/macos-linux "$name"
rm -rf dist/stage
ls -l dist
