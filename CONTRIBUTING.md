# Contributing to PatapscoAI Multi-arch Dockerfile

Thank you for your interest in contributing to this project! This repository contains a Dockerfile for building multi-architecture (arm64 and x86_64) containers with Axolotl and vLLM.

## Overview

This GitHub repository is a mirror of our internal GitLab repository. We welcome contributions from the community and have established a workflow to ensure quality while maintaining our internal development processes.

## Contribution Workflow

### 1. Fork and Clone
- Fork this repository on GitHub
- Clone your fork locally:
  ```bash
  git clone https://github.com/your-username/repository-name.git
  cd repository-name
  ```

### 2. Create a Feature Branch
- Create a new branch for your changes:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Use descriptive branch names that clearly indicate the purpose of your changes

### 3. Make Your Changes
- Implement your improvements or fixes
- Test your changes locally to ensure they work as expected
- Follow the existing code style and conventions
- Ensure your changes work for both arm64 and x86_64 architectures

### 4. Commit Your Changes
- Write clear, descriptive commit messages
- Follow conventional commit format when possible:
  ```
  type: brief description
  
  More detailed explanation if needed
  ```

### 5. Submit a Pull Request
- Push your branch to your fork:
  ```bash
  git push origin feature/your-feature-name
  ```
- Open a Pull Request against the main branch of this repository
- Provide a clear description of your changes and their purpose
- Reference any related issues

## Review and Merge Process

1. **Community Review**: Other contributors and maintainers will review your PR on GitHub
2. **Internal Testing**: Our team will test your changes in our internal GitLab environment
3. **Internal Merge**: If tests pass, we'll merge the changes in our internal GitLab repository
4. **GitHub Sync**: The changes will then be pushed to this GitHub repository

Please note that there may be a delay between PR approval and the changes appearing in the GitHub repository due to this internal testing process.

## Guidelines

### Code Quality
- Ensure all Docker builds complete successfully
- Test on both supported architectures (arm64 and x86_64)
- Follow Docker best practices for layer optimization and security
- Keep dependencies up to date and document any version requirements

### Documentation
- Update relevant documentation if your changes affect functionality
- Include comments in Dockerfile for complex operations
- Update README.md if your changes affect usage instructions

### Pull Request Guidelines
- Keep PRs focused and atomic - one feature or fix per PR
- Provide a clear description of the problem being solved
- Include any relevant context or background information
- Be responsive to feedback and questions during review

## Getting Help

If you have questions about contributing or need help with your changes:
- Open an issue on GitHub for general questions
- Use draft PRs for work-in-progress contributions where you'd like early feedback
- Check existing issues and PRs to see if your question has been addressed

## Types of Contributions Welcome

We welcome various types of contributions, including:
- Bug fixes and performance improvements
- Documentation improvements
- Dependency updates
- Build optimizations
- Support for additional architectures or platforms
- Security enhancements

## Code of Conduct

We expect all contributors to be respectful and professional in their interactions. Please be patient as we work through the review process, and provide constructive feedback when reviewing others' contributions.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

---

Thank you for contributing to PatapscoAI! Your efforts help make this project better for everyone.
