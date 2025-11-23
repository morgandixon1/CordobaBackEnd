# Contributing to Cordoba

Thank you for your interest in contributing to Cordoba! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/CordobaBackEnd.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up your development environment (see README.md)

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r Docker/requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```

3. Add your API keys to `.env`:
   - Get Mapbox token from https://account.mapbox.com/access-tokens/
   - Get OpenTopoData key from https://www.opentopodata.org/

## Making Changes

1. Make your changes in your feature branch
2. Test your changes thoroughly
3. Ensure code follows the existing style
4. Update documentation if needed

## Submitting Changes

1. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request from your fork to the main repository

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass (if applicable)
- Update documentation for new features
- Keep changes focused and atomic

## Code Style

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and concise

## Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Any relevant error messages or logs

## Feature Requests

We welcome feature requests! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation suggestions

## Security Issues

**DO NOT** open public issues for security vulnerabilities. Instead:
- Email the maintainers directly
- Provide details about the vulnerability
- Allow time for a fix before public disclosure

See SECURITY.md for more details.

## Questions?

Feel free to open an issue for questions or start a discussion in the GitHub Discussions tab.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Cordoba! 🎉
