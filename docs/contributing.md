# Contributing to SpaceMining

Thank you for your interest in contributing to the SpaceMining project! This guide will help you get started with the contribution process, from setting up your development environment to submitting pull requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

## Code of Conduct

In the interest of fostering an open and welcoming environment, we expect all contributors to be respectful and considerate of others. By participating in this project, you agree to:

- Be respectful of different viewpoints and experiences.
- Gracefully accept constructive criticism.
- Focus on what is best for the community.
- Show empathy towards other community members.

## How Can I Contribute?

There are many ways to contribute to SpaceMining, including:

- **Reporting Bugs**: If you find a bug, please open an issue with a detailed description, including steps to reproduce the issue.
- **Suggesting Enhancements**: Propose new features or improvements by opening an issue with the tag 'enhancement'.
- **Code Contributions**: Implement new features, fix bugs, or improve existing code by submitting pull requests.
- **Documentation**: Help improve project documentation by correcting typos, clarifying existing content, or adding new guides.
- **Testing**: Write or improve test cases to ensure the stability and reliability of the codebase.

## Development Setup

To set up the development environment for SpaceMining:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/reveurmichael/space_mining.git
   cd space_mining
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install in Development Mode**:
   ```bash
   pip install -e '.[dev]'
   ```
   This installs the project in editable mode along with development dependencies like `pytest`, `black`, and `isort`.

4. **Verify Setup**:
   Run the tests to ensure everything is set up correctly:
   ```bash
   pytest
   ```

## Coding Guidelines

We aim to maintain a high-quality, consistent codebase. Please follow these guidelines when contributing code:

- **Style Guide**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Use `black` for automatic code formatting and `isort` for import sorting:
  ```bash
  black .
  isort .
  ```
- **Type Hints**: Use type hints where possible to improve code readability and maintainability.
- **Documentation**: Document all public modules, functions, classes, and methods using docstrings in the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Commit Messages**: Write clear, concise commit messages that describe the purpose of the changes. Follow the [conventional commits](https://www.conventionalcommits.org/) format if possible (e.g., `feat: add new reward function`, `fix: correct energy depletion bug`).

## Pull Request Process

1. **Create a Branch**:
   Create a branch with a descriptive name related to the issue or feature you're working on:
   ```bash
   git checkout -b feat/add-multi-agent-support
   ```

2. **Make Your Changes**:
   Implement your changes, commit them with meaningful messages, and ensure your code follows the coding guidelines.

3. **Run Tests**:
   Before submitting, run the test suite to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. **Update Documentation**:
   If your changes affect the user experience or API, update the relevant documentation in the `./docs/` folder or inline docstrings.

5. **Submit Your Pull Request**:
   Push your branch to the repository and create a pull request (PR) against the `main` branch:
   ```bash
   git push origin feat/add-multi-agent-support
   ```
   - Fill in the PR template with a clear description of your changes, referencing related issues if applicable.
   - Ensure your PR passes all CI checks (if set up).

6. **Code Review**:
   Maintainers will review your PR. Address any feedback by making necessary changes and updating your branch:
   ```bash
   git commit -am 'address review comments'
   git push
   ```

7. **Merge**:
   Once approved, your PR will be merged into the `main` branch.

## Testing Guidelines

Tests are crucial for maintaining the reliability of SpaceMining. When contributing new features or bug fixes:

- **Write Tests**: Add tests for new functionality in the `./tests/` folder. Follow the existing test structure, using `pytest`.
- **Run Tests**: Ensure all tests pass before submitting a PR:
  ```bash
  pytest
  ```
- **Coverage**: Aim for high test coverage, especially for critical components like environment logic and reward functions.

Example of a simple test file structure:

```python
# tests/test_env_step.py
def test_environment_step():
    from space_mining import make_env
    env = make_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    env.close()
```

## Documentation Guidelines

Good documentation helps users and contributors understand and use SpaceMining effectively:

- **Inline Documentation**: Use docstrings for all public APIs (classes, methods, functions) following the Google style.
- **User Guides**: Update or add markdown files in `./docs/` for tutorials, guides, or detailed explanations of features.
- **API Reference**: If using Sphinx, ensure API documentation is generated correctly from docstrings.
- **README Updates**: If your contribution significantly changes usage or setup, update the main `README.md` with relevant information.

## Community

- **GitHub Issues**: For bug reports, feature requests, and general questions.

Thank you for contributing to SpaceMining and helping make it better! If you have any questions or need assistance during the contribution process, don't hesitate to ask.
