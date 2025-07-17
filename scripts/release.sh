#!/usr/bin/env bash
# this_file: scripts/release.sh

set -e

VERSION=${1:-}
if [ -z "$VERSION" ]; then
    echo "❌ Usage: $0 <version>"
    echo "   Example: $0 1.2.3"
    exit 1
fi

echo "🚀 Creating release for version $VERSION..."

# Validate version format
if ! echo "$VERSION" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\-\.]+)?$' > /dev/null; then
    echo "❌ Invalid version format. Use semantic versioning (e.g., 1.2.3)"
    exit 1
fi

# Check if we're on main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "❌ Release must be created from main branch. Current branch: $BRANCH"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Run tests
echo "🧪 Running tests..."
./scripts/test.sh

# Build package
echo "🏗️ Building package..."
./scripts/build.sh

# Create and push tag
echo "🏷️ Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"
git push origin "v$VERSION"

echo "✅ Release v$VERSION created successfully!"
echo "📦 Package built and ready for deployment"
echo "🏷️ Tag v$VERSION pushed to repository"
echo ""
echo "📋 Next steps:"
echo "   1. GitHub Actions will automatically create a release"
echo "   2. Built packages are available in dist/"
echo "   3. To publish to PyPI: python3 -m twine upload dist/*"