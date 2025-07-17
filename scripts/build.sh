#!/usr/bin/env bash
# this_file: scripts/build.sh

set -e

echo "🔧 Building vidkompy..."

# Clean previous builds
echo "📦 Cleaning previous builds..."
rm -rf dist/ build/ src/vidkompy.egg-info/

# Install/upgrade build dependencies
echo "📦 Installing build dependencies..."
python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages build hatch

# Build the package
echo "🏗️ Building package..."
python3 -m build

# Verify build
echo "✅ Verifying build..."
if [ -d "dist/" ] && [ "$(ls -1 dist/ | wc -l)" -gt 0 ]; then
    echo "✅ Build successful!"
    echo "📦 Built packages:"
    ls -la dist/
else
    echo "❌ Build failed - no packages found in dist/"
    exit 1
fi

echo "🎉 Build complete!"