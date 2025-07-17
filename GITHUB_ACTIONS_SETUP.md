# GitHub Actions Setup Guide

## Overview

This guide explains how to set up the GitHub Actions workflows for automated CI/CD. The workflow files are provided in the `github-workflows/` directory but need to be manually moved to `.github/workflows/` due to permission restrictions.

## Setup Steps

### 1. Create the workflows directory

```bash
mkdir -p .github/workflows
```

### 2. Copy the workflow files

```bash
cp github-workflows/ci.yml .github/workflows/
cp github-workflows/release.yml .github/workflows/
```

### 3. Remove the temporary directory

```bash
rm -rf github-workflows/
```

### 4. Commit and push the workflows

```bash
git add .github/workflows/
git commit -m "feat: add GitHub Actions CI/CD workflows"
git push
```

## Workflow Files

### `ci.yml` - Continuous Integration
- **Triggers**: Push to main branch, pull requests
- **Matrix**: Ubuntu/Windows/macOS × Python 3.10/3.11/3.12
- **Actions**: Install FFmpeg, run tests, linting, type checking, coverage

### `release.yml` - Release Automation
- **Triggers**: Git tags matching `v*` (e.g., `v1.2.3`)
- **Actions**: 
  - Test on all platforms
  - Build Python packages
  - Build platform-specific binaries
  - Create GitHub releases
  - Publish to PyPI

## Required Secrets and Settings

### PyPI Publishing (Optional)
For automated PyPI publishing, set up trusted publishing:

1. Go to [PyPI Trusted Publishing](https://pypi.org/manage/account/publishing/)
2. Add a new trusted publisher:
   - **Owner**: Your GitHub username
   - **Repository**: vidkompy
   - **Workflow**: release.yml
   - **Environment**: pypi

### Codecov (Optional)
For coverage reporting:

1. Go to [Codecov](https://codecov.io/)
2. Add your repository
3. Add `CODECOV_TOKEN` secret in repository settings

## Usage

### Continuous Integration
CI runs automatically on:
- Push to main branch
- Pull requests to main branch

### Releases
Create releases by pushing git tags:

```bash
# Using the release script
./scripts/release.sh 1.2.3

# Or manually
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3
```

## Troubleshooting

### Permission Issues
If you see permission errors about workflow files, ensure:
1. You have admin access to the repository
2. GitHub Actions are enabled in repository settings
3. The workflows are in the correct `.github/workflows/` directory

### FFmpeg Installation Failures
If FFmpeg installation fails:
- Check the package manager commands for your platform
- Consider using pre-built FFmpeg binaries
- Update the workflow to use different installation methods

### Build Failures
If binary builds fail:
- Check PyInstaller configuration in `vidkompy.spec`
- Ensure all dependencies are properly included
- Test binary builds locally first

## Testing Workflows

### Test CI Workflow
1. Create a feature branch
2. Make changes and push
3. Create a pull request
4. Check the CI status

### Test Release Workflow
1. Create a test tag: `git tag v0.0.1-test`
2. Push the tag: `git push origin v0.0.1-test`
3. Check GitHub Actions tab for workflow execution
4. Verify artifacts are created

## Customization

### Modify Python Versions
Edit the matrix in both workflows:
```yaml
python-version: ["3.10", "3.11", "3.12"]
```

### Modify Platforms
Edit the matrix in both workflows:
```yaml
os: [ubuntu-latest, windows-latest, macos-latest]
```

### Add Custom Steps
Add new steps to the workflows as needed:
```yaml
- name: Custom Step
  run: |
    echo "Custom command here"
```

## Best Practices

1. **Test locally first**: Always test scripts locally before pushing
2. **Use specific versions**: Pin action versions for reproducibility
3. **Secure secrets**: Use GitHub secrets for sensitive information
4. **Monitor workflows**: Check workflow runs regularly
5. **Update dependencies**: Keep actions and dependencies updated

## Support

For issues with the workflows:
1. Check the GitHub Actions logs
2. Verify all dependencies are installed
3. Test the scripts locally
4. Review the workflow configuration
5. Check repository permissions and settings