# Publishing Documentation

This document explains how the automatic publishing workflows are set up and how to maintain them.

## PyPI Publishing

### How It Works

The repository is configured to automatically publish to PyPI whenever code is pushed to the `main` branch. The workflow uses PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) feature, which is more secure than using API tokens.

### Setup Steps

1. **Configure PyPI Trusted Publishing** (one-time setup):
   - Go to https://pypi.org/manage/account/publishing/
   - Log in with your PyPI account (you must be an owner or maintainer of the package)
   - Click "Add a new pending publisher"
   - Fill in:
     - PyPI Project Name: `meegflow`
     - Owner: `Picinic-DoC` (the GitHub organization/user)
     - Repository name: `meegflow`
     - Workflow name: `publish-pypi.yml`
     - Environment name: (leave blank)
   - Click "Add"

2. **First Publication**:
   - The first time you push to `main`, the workflow will fail if the package doesn't exist on PyPI yet
   - You may need to manually publish the first version using `python -m build` and `twine upload dist/*`
   - Alternatively, PyPI will create the project automatically when the trusted publisher credentials are configured

3. **Versioning**:
   - The version is defined in `setup.py` (currently `0.1.0`)
   - Before pushing to `main`, update the version number in `setup.py`
   - The workflow will skip publishing if a version already exists (using `skip-existing: true`)

### Workflow Details

The workflow (`.github/workflows/publish-pypi.yml`) performs these steps:
1. Checks out the code
2. Sets up Python 3.11
3. Installs build tools (`build` and `twine`)
4. Builds the distribution packages (wheel and source distribution)
5. Validates the distribution with `twine check`
6. Publishes to PyPI using OIDC authentication

## Conda-Forge Publishing

### How It Works

Conda-forge uses a "feedstock" repository system. After your package is on PyPI, you can submit a recipe to conda-forge, and they will create a separate feedstock repository that automatically builds and publishes conda packages.

### Setup Steps

1. **Initial Recipe Submission** (one-time setup):
   - Fork the staged-recipes repository: https://github.com/conda-forge/staged-recipes
   - Create a new branch in your fork
   - Copy the recipe from `.conda-recipe/meta.yaml` to `recipes/meegflow/meta.yaml` in the staged-recipes repo
   - Update the `sha256` field in `meta.yaml` with the hash of the PyPI source tarball:
     ```bash
     # Get the sha256 from PyPI after first upload
     curl -L https://pypi.io/packages/source/m/meegflow/meegflow-0.1.0.tar.gz | shasum -a 256
     ```
   - Update the `recipe-maintainers` section with your GitHub username(s)
   - Commit and push to your fork
   - Open a Pull Request to conda-forge/staged-recipes
   - Wait for the conda-forge bot to review and merge (usually takes a few days)

2. **After Feedstock Creation**:
   - Once merged, conda-forge will create a new repository: `conda-forge/meegflow-feedstock`
   - The feedstock repository will have an automated bot that watches PyPI
   - When you publish a new version to PyPI, the bot will automatically create a PR to update the conda package
   - You (as a maintainer) will need to review and merge these automated PRs

3. **Adding More Maintainers**:
   - Edit the `recipe/meta.yaml` file in the feedstock repository
   - Add GitHub usernames to the `extra.recipe-maintainers` section
   - Submit a PR

### Recipe Details

The conda recipe (`.conda-recipe/meta.yaml`) includes:
- Package metadata (name, version)
- Source location (PyPI)
- Build instructions (noarch python package)
- Dependencies (runtime and build requirements)
- Tests (import checks and CLI help command)
- About section (license, description, links)
- Maintainers list

## Version Management

### Recommended Versioning Strategy

1. Use semantic versioning (MAJOR.MINOR.PATCH):
   - MAJOR: Breaking changes
   - MINOR: New features (backwards compatible)
   - PATCH: Bug fixes

2. Before pushing to `main`:
   - Update version in `pyproject.toml`
   - Update version in `setup.py` (optional, but recommended for backwards compatibility)
   - Update version in `.conda-recipe/meta.yaml` (for reference)
   - Commit the version bump
   - Push to `main`

3. Consider using git tags:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

## Testing Before Publishing

To test the package builds correctly before pushing to `main`:

```bash
# Test building the package
python -m pip install build twine
python -m build

# Check the distribution
twine check dist/*

# Test installing locally
pip install dist/meegflow-*.whl

# Test the CLI
meegflow --help
```

## Troubleshooting

### PyPI Publishing Fails

1. **"403 Forbidden" error**:
   - Check that Trusted Publishing is configured correctly on PyPI
   - Verify the workflow name matches exactly: `publish-pypi.yml`
   - Ensure you have owner/maintainer permissions on PyPI

2. **"File already exists" error**:
   - You're trying to upload a version that already exists
   - Update the version number in `setup.py`
   - The workflow uses `skip-existing: true` to avoid this, but manual uploads may still fail

3. **Build fails**:
   - Check that all files are included properly
   - Verify `pyproject.toml` and `setup.py` are correct
   - Test locally with `python -m build`

4. **"twine check" warnings about metadata 2.4**:
   - Current versions of twine may show warnings about `license-file` or `license-expression` fields
   - This is a known compatibility issue with metadata format 2.4
   - PyPI itself accepts metadata 2.4, so these warnings can be safely ignored
   - The GitHub Actions workflow doesn't use `twine check` before publishing

### Conda-Forge Issues

1. **Recipe doesn't build**:
   - Check that all dependencies are available on conda-forge
   - Test the recipe locally using `conda-build`
   - Check the conda-forge documentation: https://conda-forge.org/docs/maintainer/

2. **Bot doesn't auto-update**:
   - The bot checks PyPI periodically (may take a few hours)
   - You can manually trigger an update by opening an issue in the feedstock repo
   - Check the feedstock CI logs for errors

## Security Notes

- The PyPI workflow uses **Trusted Publishing** (OIDC), which is more secure than API tokens
- No secrets need to be stored in GitHub
- The workflow has minimal permissions (`id-token: write`, `contents: read`)
- Always review automated PRs before merging
- Keep dependencies up to date to avoid security vulnerabilities

## References

- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- Conda-Forge Documentation: https://conda-forge.org/docs/
- Python Packaging Guide: https://packaging.python.org/
