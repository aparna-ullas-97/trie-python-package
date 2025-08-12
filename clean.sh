#!/usr/bin/env bash
# Clean old Python build artifacts for the TRIE SDK

echo "🧹 Cleaning old build artifacts..."

# Enable nullglob so patterns that don't match expand to nothing
shopt -s nullglob 2>/dev/null || setopt NULL_GLOB 2>/dev/null

# Remove build directories
rm -rf build dist

# Remove all egg-info directories
rm -rf *.egg-info src/*.egg-info

# Disable nullglob again
shopt -u nullglob 2>/dev/null || unsetopt NULL_GLOB 2>/dev/null

echo "✅ Clean complete."