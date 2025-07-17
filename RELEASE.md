# Release Guide for vidkompy

## Overview

This document describes the git-tag-based semversioning and release system for vidkompy, including local development scripts and automated GitHub Actions.

## Versioning System

### Git-Tag-Based Semversioning

The project uses `hatch-vcs` for automatic version management based on git tags:

- **Version format**: Follows semantic versioning (e.g., `1.2.3`)
- **Automatic versioning**: Version is automatically derived from git tags
- **Development versions**: Between releases, versions include commit info (e.g., `1.2.3.dev0+abc123`)

### Creating Releases

#### Local Release Process

1. **Prepare for release**:
   ```bash
   # Ensure you're on main branch
   git checkout main
   git pull origin main
   
   # Run tests to ensure everything works
   ./scripts/test.sh
   ```

2. **Create release using the script**:
   ```bash
   ./scripts/release.sh 1.2.3
   ```

   This script will:
   - Validate the version format
   - Ensure you're on the main branch
   - Run all tests
   - Build the package
   - Create and push a git tag
   - Trigger automated GitHub Actions

#### Manual Release Process

If you prefer manual control:

1. **Test everything**:
   ```bash
   ./scripts/test.sh
   ```

2. **Create and push tag**:
   ```bash
   git tag -a v1.2.3 -m "Release version 1.2.3"
   git push origin v1.2.3
   ```

3. **GitHub Actions will automatically**:
   - Run tests on all platforms
   - Build Python packages
   - Build binary executables
   - Create GitHub release
   - Publish to PyPI

## Local Development Scripts

### Available Scripts

| Script | Purpose |
|--------|---------|
| `./scripts/dev-setup.sh` | Set up development environment |
| `./scripts/test.sh` | Run comprehensive tests |
| `./scripts/build.sh` | Build Python packages |
| `./scripts/build-binary.sh` | Build platform-specific binary |
| `./scripts/release.sh X.Y.Z` | Create complete release |

### Development Workflow

1. **Initial setup**:
   ```bash
   ./scripts/dev-setup.sh
   ```

2. **During development**:
   ```bash
   # Run tests frequently
   ./scripts/test.sh
   
   # Build and test locally
   ./scripts/build.sh
   ```

3. **Before releasing**:
   ```bash
   # Final test run
   ./scripts/test.sh
   
   # Create release
   ./scripts/release.sh 1.2.3
   ```

## GitHub Actions CI/CD

### Setup Required

**⚠️ Important**: GitHub Actions workflows require manual setup due to permission restrictions.

See `GITHUB_ACTIONS_SETUP.md` for complete setup instructions.

### Continuous Integration (CI)

**Trigger**: Push to main branch or pull request

**Actions**: 
- Tests on Ubuntu, Windows, macOS
- Python versions: 3.10, 3.11, 3.12
- Linting with ruff
- Type checking with mypy
- Code coverage reporting

### Release Workflow

**Trigger**: Push tag matching `v*` (e.g., `v1.2.3`)

**Actions**:
1. **Test**: Run full test suite on all platforms
2. **Build Python Package**: Create wheel and source distribution
3. **Build Binaries**: Create standalone executables for:
   - Linux (`vidkompy-linux`)
   - Windows (`vidkompy-windows.exe`)
   - macOS (`vidkompy-macos`)
4. **Create GitHub Release**: Automatically with generated notes
5. **Publish to PyPI**: Using trusted publishing (no API key needed)

### Setting Up GitHub Actions

#### Manual Setup Steps

1. **Copy workflow files**:
   ```bash
   mkdir -p .github/workflows
   cp github-workflows/ci.yml .github/workflows/
   cp github-workflows/release.yml .github/workflows/
   ```

2. **Commit and push**:
   ```bash
   git add .github/workflows/
   git commit -m "feat: add GitHub Actions CI/CD workflows"
   git push
   ```

#### Required Repository Secrets

1. **PyPI Publishing**: Set up trusted publishing in PyPI settings
2. **GitHub Token**: Automatically provided by GitHub

#### Optional Configuration

- **Codecov**: Add `CODECOV_TOKEN` for coverage reporting
- **Custom PyPI**: Modify workflow if using private PyPI instance

## Binary Distribution

### Standalone Executables

The release process creates standalone executables that don't require Python installation:

- **Linux**: `vidkompy-linux`
- **Windows**: `vidkompy-windows.exe`
- **macOS**: `vidkompy-macos`

### Binary Features

- **Self-contained**: Includes all dependencies
- **No Python required**: Can run on systems without Python
- **Full functionality**: Same features as pip-installed version
- **FFmpeg dependency**: Still requires FFmpeg to be installed separately

### Building Binaries Locally

```bash
# Build for current platform
./scripts/build-binary.sh

# Binary will be in dist/vidkompy-<platform>
```

## Installation Methods

### For End Users

1. **PyPI (Recommended)**:
   ```bash
   pip install vidkompy
   ```

2. **GitHub Releases**:
   - Download from GitHub releases page
   - Choose appropriate binary for your platform

3. **From Source**:
   ```bash
   git clone https://github.com/twardoch/vidkompy.git
   cd vidkompy
   pip install -e .
   ```

### For Developers

```bash
git clone https://github.com/twardoch/vidkompy.git
cd vidkompy
./scripts/dev-setup.sh
```

## Testing

### Test Categories

1. **Unit Tests**: Core functionality testing
2. **Integration Tests**: CLI and workflow testing
3. **Import Tests**: Module import validation
4. **Version Tests**: Versioning system validation

### Test Commands

```bash
# Run all tests
./scripts/test.sh

# Run specific test categories
python -m pytest tests/test_core_functionality.py -v
python -m pytest tests/test_cli.py -v
python -m pytest tests/test_version.py -v

# Run with coverage
python -m pytest --cov=src/vidkompy tests/
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in PATH
2. **Import errors**: Run `./scripts/dev-setup.sh` to install dependencies
3. **Version issues**: Delete `src/vidkompy/__version__.py` and rebuild
4. **Binary build fails**: Ensure all dependencies are installed

### Debug Commands

```bash
# Check version
python -c "import vidkompy; print(vidkompy.__version__)"

# Test CLI
python -m vidkompy --version

# Check FFmpeg
ffmpeg -version
```

## Release Checklist

- [ ] All tests passing locally
- [ ] Code formatted and linted
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number follows semver
- [ ] On main branch with clean working directory
- [ ] FFmpeg dependency documented
- [ ] GitHub Actions secrets configured
- [ ] PyPI trusted publishing set up

## Support

For issues with the release system:
1. Check GitHub Actions logs
2. Verify all dependencies are installed
3. Ensure FFmpeg is available
4. Check PyPI trusted publishing configuration