Changelog
=========

All notable changes to this project will be documented in this file.

0.1.1 - 2025-09-13
------------------
- Fix: Lazy-import OpenAI SDK in agent to avoid import-time failures when the SDK is absent.
- Fix: Remove hard pandas timestamp dependency in bathymetry, use `np.datetime64('now')`.
- Fix: Pin dependency ranges for Python 3.9 to avoid NumPy 2.x / SciPy / xarray breakages.
- Chore: Update .gitignore and untrack virtualenv/build artifacts.
- Docs: Update README version and release notes.

0.1.0 - 2025-09-12
------------------
- Initial public package structure with CLI, FastAPI app, analysis modules, ML utilities, monitoring, remediation, and tests.
