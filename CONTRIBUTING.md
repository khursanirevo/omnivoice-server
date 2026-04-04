# Contributing to OmniVoice Server

Thank you for your interest in contributing to OmniVoice Server! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to matthew.ngo1114@gmail.com.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, device)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- Clear, descriptive title
- Detailed description of the proposed functionality
- Rationale for why this enhancement would be useful
- Possible implementation approach (optional)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding style**:
   - PEP 8 for Python code
   - Type hints on all function signatures
   - Docstrings for public functions
3. **Write tests** for new functionality
4. **Ensure all tests pass**: `pytest tests/`
5. **Run linters**: `ruff check .` and `mypy omnivoice_server/`
6. **Update documentation** if needed
7. **Write clear commit messages** following conventional commits format

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/omnivoice-server.git
cd omnivoice-server

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linters
ruff check .
mypy omnivoice_server/
```

### Coding Standards

- **Immutability**: Prefer immutable data structures (frozen dataclasses)
- **Error handling**: Handle errors explicitly, provide user-friendly messages
- **Type safety**: Use type hints, avoid `Any` when possible
- **Testing**: Aim for 80%+ test coverage
- **Documentation**: Update README and docstrings for public APIs

### Commit Message Format

```
<type>: <description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

Examples:
- `feat: add support for custom voice profiles`
- `fix: resolve memory leak in inference service`
- `docs: update API endpoint documentation`

## Project Structure

```
omnivoice-server/
├── omnivoice_server/     # Main package
│   ├── routers/          # FastAPI route handlers
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── benchmarks/           # Performance benchmarks
├── examples/             # Usage examples
└── docs/                 # Documentation
```

## Questions?

Feel free to open an issue for questions or reach out to matthew.ngo1114@gmail.com.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
