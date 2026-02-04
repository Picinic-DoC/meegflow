# Automatic Publishing Setup Summary

This document provides a summary of the automatic publishing implementation for the MEEGFlow repository.

## What Was Implemented

### 1. PyPI Publishing Workflows

Two GitHub Actions workflows were created to automate publishing to PyPI:

#### A. Push-to-Main Workflow (`.github/workflows/publish-pypi.yml`)
- **Trigger**: Automatically runs on every push to the `main` branch
- **Purpose**: Continuous deployment for rapid iteration
- **Process**:
  1. Builds Python package (wheel and source distribution)
  2. Validates distribution with twine
  3. Publishes to PyPI using OIDC trusted publishing
- **Feature**: Uses `skip-existing: true` to avoid failures on duplicate versions

#### B. Tag-Based Release Workflow (`.github/workflows/publish-release.yml`) - **RECOMMENDED**
- **Trigger**: Runs when a version tag (e.g., `v0.1.0`) is pushed
- **Purpose**: Controlled release process for production deployments
- **Process**:
  1. Builds Python package
  2. Publishes to PyPI using OIDC trusted publishing
  3. Creates a GitHub Release with artifacts and auto-generated release notes
- **Benefits**:
  - More deliberate release process
  - GitHub Releases for easy version tracking
  - Downloadable artifacts attached to each release

### 2. Package Configuration

#### Modern Python Packaging Setup
- **`pyproject.toml`**: Modern package metadata following PEP 621
  - Package name, version, description
  - Dependencies from requirements.txt
  - Entry points for CLI command
  - License and author information
  - Python version requirements

- **`setup.py`**: Updated for compatibility
  - Fixed repository URL to point to `Picinic-DoC/meegflow`
  - Added license field

- **`MANIFEST.in`**: Ensures important files are included in distributions
  - LICENSE
  - README.md
  - requirements.txt
  - Config files (configs/*.yaml)

- **`LICENSE`**: MIT License file added

### 3. Conda-Forge Recipe

Created `.conda-recipe/meta.yaml` with:
- Package metadata and source configuration
- Build instructions for noarch Python package
- Complete dependency specifications
- Import and CLI tests
- License and documentation links
- Maintainer placeholders

### 4. Documentation

#### PUBLISHING.md - Comprehensive Publishing Guide
- **Setup Instructions**:
  - How to configure PyPI Trusted Publishing
  - Steps for both workflow types
  - First-time publication guidance

- **Publishing Strategies**:
  - Detailed comparison of push-to-main vs tag-based approaches
  - Step-by-step release process for each method
  - How to disable unused workflows

- **Conda-Forge Instructions**:
  - How to submit initial recipe to staged-recipes
  - Feedstock maintenance
  - Adding additional maintainers

- **Version Management**:
  - Semantic versioning recommendations
  - Version update procedures for both workflows
  - File locations for version numbers

- **Testing Guidelines**:
  - Local build testing
  - Package installation verification
  - CLI testing

- **Troubleshooting**:
  - Common PyPI publishing errors and solutions
  - Conda-forge issues
  - Known metadata 2.4 compatibility notes

#### README.md Updates
- Added PyPI installation as Option 1 (recommended)
- Updated repository URLs from old repo to `Picinic-DoC/meegflow`
- Reorganized installation options for clarity

#### SETUP_SUMMARY.md (This File)
- Implementation overview
- Configuration details
- Next steps

## Security Features

- **Trusted Publishing (OIDC)**: No API tokens needed, more secure
- **Minimal Permissions**: Workflows use least-privilege permissions
- **No Secrets Required**: Authentication handled via GitHub's OIDC
- **Skip Existing**: Prevents accidental overwrites of published versions

## Files Created/Modified

### New Files
- `.github/workflows/publish-pypi.yml` - Push-to-main publishing workflow
- `.github/workflows/publish-release.yml` - Tag-based release workflow
- `pyproject.toml` - Modern Python package metadata
- `MANIFEST.in` - Distribution file inclusion rules
- `LICENSE` - MIT license
- `.conda-recipe/meta.yaml` - Conda-forge recipe template
- `PUBLISHING.md` - Complete publishing documentation
- `SETUP_SUMMARY.md` - This summary document

### Modified Files
- `setup.py` - Updated repository URL and added license
- `README.md` - Added PyPI installation, fixed URLs

## Next Steps for Repository Maintainers

### Immediate Setup (Required)

1. **Configure PyPI Trusted Publishing**:
   - Create PyPI account if needed
   - Go to https://pypi.org/manage/account/publishing/
   - Add trusted publishers for both workflows (or just the one you'll use)
   - Follow the specific instructions in PUBLISHING.md

2. **Choose Your Publishing Strategy**:
   - **Tag-based** (recommended): More controlled, includes GitHub Releases
   - **Push-to-main**: Automatic on every merge
   - Disable the unused workflow if you only want one

3. **First Publication**:
   - May need to manually publish v0.1.0 with `twine upload`
   - Or let PyPI auto-create on first workflow run (if trusted publisher configured)

### Optional Setup (Recommended)

4. **Conda-Forge Submission** (after first PyPI release):
   - Fork conda-forge/staged-recipes
   - Copy `.conda-recipe/meta.yaml` to `recipes/meegflow/meta.yaml`
   - Update sha256 hash from PyPI
   - Add your GitHub username as maintainer
   - Submit PR to staged-recipes
   - Wait for review and merge (usually a few days)

5. **Set Up Branch Protection** (optional):
   - Require PR reviews before merging to main
   - Prevent accidental direct pushes that trigger publishing

6. **Create First Release**:
   ```bash
   # Update version in pyproject.toml to 0.1.0
   git add pyproject.toml
   git commit -m "Release version 0.1.0"
   git push origin main
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

## Testing the Setup

Before the first real release, you can test locally:

```bash
# Build the package
python -m pip install build twine
python -m build

# Check the distribution (may show metadata 2.4 warnings - these are safe to ignore)
twine check dist/*

# Test installation locally
pip install dist/meegflow-*.whl
meegflow --help
```

## Support and References

- **PyPI Trusted Publishing**: https://docs.pypi.org/trusted-publishers/
- **Conda-Forge Docs**: https://conda-forge.org/docs/
- **Python Packaging Guide**: https://packaging.python.org/
- **GitHub Actions**: https://docs.github.com/en/actions

## Notes

- The package uses metadata format 2.4, which is accepted by PyPI but may show warnings with older `twine` versions
- Both workflows use `skip-existing: true` to prevent errors on duplicate versions
- Version must be updated in `pyproject.toml` for new releases
- The conda recipe will need the PyPI sha256 hash after first upload
