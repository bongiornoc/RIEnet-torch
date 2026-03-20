# Releasing `rienet-torch`

## Recommended public names

- GitHub repository: `rienet-torch`
- PyPI distribution: `rienet-torch`
- Python import package: `rienet_torch`

If `rienet-torch` is not available on PyPI, the clean fallback is `rienet-pytorch`.

## One-time setup

1. Create the repository on GitHub.
2. Create the package on PyPI with the chosen distribution name.
3. In PyPI, configure a Trusted Publisher for this repository:
   - Owner: your GitHub user or organization
   - Repository: `rienet-torch`
   - Workflow: `publish.yml`
   - Environment: `pypi`
4. Optionally configure another Trusted Publisher on TestPyPI:
   - Workflow: `publish-testpypi.yml`
   - Environment: `testpypi`
5. In GitHub repository settings, create the environments `pypi` and `testpypi`.

## Release flow

1. Bump the version in `src/rienet_torch/version.py`.
2. Run the local validation commands:

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not local_only" tests
python -m build
python -m twine check dist/*
```

3. Commit the version bump.
4. Push the commit to `main`.

5. GitHub Actions will build and publish automatically to PyPI when the
   version bump lands on `main`.
6. If you need to retry a release without another commit, run the
   `Publish to PyPI` workflow manually from the GitHub Actions UI.

## TestPyPI flow

Use the `Publish to TestPyPI` workflow from the GitHub Actions UI when you want
to validate the package metadata and installation flow before a real release.
