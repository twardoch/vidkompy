#!/usr/bin/env bash
# this_file: scripts/build-binary.sh

set -e

echo "🔧 Building vidkompy binary..."

# Determine platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

echo "📦 Building for platform: $PLATFORM"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/vidkompy-* build/

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages -e ".[dev]"

# Build binary
echo "🏗️ Building binary..."
if [[ "$PLATFORM" == "windows" ]]; then
    pyinstaller --clean --onefile --name "vidkompy-${PLATFORM}.exe" vidkompy.spec
else
    pyinstaller --clean --onefile --name "vidkompy-${PLATFORM}" vidkompy.spec
fi

# Verify binary
echo "✅ Verifying binary..."
if [[ "$PLATFORM" == "windows" ]]; then
    BINARY_PATH="dist/vidkompy-${PLATFORM}.exe"
else
    BINARY_PATH="dist/vidkompy-${PLATFORM}"
fi

if [ -f "$BINARY_PATH" ]; then
    echo "✅ Binary built successfully: $BINARY_PATH"
    echo "📊 Binary size: $(du -h "$BINARY_PATH" | cut -f1)"
    
    # Test the binary
    echo "🧪 Testing binary..."
    "$BINARY_PATH" --version || echo "⚠️  Binary version check failed"
else
    echo "❌ Binary build failed - file not found: $BINARY_PATH"
    exit 1
fi

echo "🎉 Binary build complete!"