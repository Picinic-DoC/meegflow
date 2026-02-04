# 🚀 Quick Start: Next Steps for Publishing

This is your quick-start guide to get automatic publishing working.

## ⚡ Immediate Action Required

### Step 1: Configure PyPI Trusted Publishing (5 minutes)

1. **Go to PyPI**: https://pypi.org/manage/account/publishing/
2. **Log in** with your PyPI account
3. **Click "Add a new pending publisher"**
4. **Fill in the form**:
   - **PyPI Project Name**: `meegflow`
   - **Owner**: `Picinic-DoC`
   - **Repository name**: `meegflow`
   - **Workflow name**: `publish-release.yml` (recommended) or `publish-pypi.yml`
   - **Environment name**: (leave blank)
5. **Click "Add"**

> 💡 **Tip**: If you want both workflows enabled, repeat step 4 with the other workflow name.

### Step 2: Choose Your Publishing Strategy

#### Option A: Tag-Based Releases (RECOMMENDED) ⭐

**Best for**: Production releases, clear version history, GitHub Releases

**How to publish a new version**:
```bash
# 1. Update version in pyproject.toml
# Edit: version = "0.2.0"

# 2. Commit and push
git add pyproject.toml
git commit -m "Release version 0.2.0"
git push origin main

# 3. Create and push tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# 4. GitHub Actions will automatically:
#    - Build the package
#    - Publish to PyPI
#    - Create a GitHub Release
```

**To disable push-to-main workflow**:
```bash
git rm .github/workflows/publish-pypi.yml
git commit -m "Use only tag-based releases"
git push origin main
```

#### Option B: Push-to-Main

**Best for**: Continuous deployment, rapid iteration

**How to publish a new version**:
```bash
# 1. Update version in pyproject.toml
# Edit: version = "0.2.0"

# 2. Commit and push to main
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin main

# 3. GitHub Actions automatically publishes to PyPI
```

**To disable tag-based workflow**:
```bash
git rm .github/workflows/publish-release.yml
git commit -m "Use only push-to-main publishing"
git push origin main
```

## 📝 Optional: Conda-Forge Setup (Later)

After your first PyPI release, you can add conda-forge support:

1. **Fork**: https://github.com/conda-forge/staged-recipes
2. **Copy recipe**: `.conda-recipe/meta.yaml` → `recipes/meegflow/meta.yaml`
3. **Get SHA256** from your PyPI release:
   ```bash
   curl -L https://pypi.io/packages/source/m/meegflow/meegflow-0.1.0.tar.gz | shasum -a 256
   ```
4. **Update** `sha256` field in recipe
5. **Add your GitHub username** to `recipe-maintainers`
6. **Submit PR** to staged-recipes
7. **Wait** for conda-forge bot review (usually 2-3 days)

## 🧪 Test Before First Release

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Test locally (optional, may show metadata warnings - safe to ignore)
twine check dist/*

# Install and test
pip install dist/meegflow-*.whl
meegflow --help
```

## ❓ FAQ

**Q: Do I need to create the PyPI project first?**
A: No! When you configure Trusted Publishing, PyPI will auto-create the project on first publish.

**Q: What if I push without updating the version?**
A: The workflow uses `skip-existing: true`, so it won't fail but also won't publish anything new.

**Q: Can I test the workflow without publishing?**
A: Not easily. However, the build step will validate your package. Consider testing in a separate PyPI test account.

**Q: Which workflow should I use?**
A: Tag-based is recommended for most projects as it gives you more control and creates nice GitHub Releases.

## 📞 Need Help?

- See **PUBLISHING.md** for detailed documentation
- See **SETUP_SUMMARY.md** for implementation details
- Check **Troubleshooting** section in PUBLISHING.md

## ✅ Ready to Publish!

Once you've configured PyPI (Step 1) and chosen your strategy (Step 2), you're ready to publish your first release!

For tag-based (recommended):
```bash
git tag -a v0.1.0 -m "First release"
git push origin v0.1.0
```

Or for push-to-main:
```bash
# Just push to main!
git push origin main
```

The GitHub Actions workflow will handle the rest! 🎉
