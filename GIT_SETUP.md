# Git Configuration Guide

## Setting Up Your Git Identity

To ensure all commits appear as your own work, run these commands:

```bash
# Set your name
git config --global user.name "Dawit L. Gulta"

# Set your email
git config --global user.email "dawit.lambebo@gmail.com"

# Optional: Set your preferred editor
git config --global core.editor "code --wait"

# Verify your configuration
git config --global --list
```

## Creating Your First Commit

```bash
# Add all files to staging
git add .

# Create your initial commit with your message
git commit -m "Initial project setup for medical computer vision"

# Check commit author
git log --oneline -1
```

## .gitignore Configuration

The project already includes a comprehensive .gitignore that excludes:
- Large datasets (`data/raw/`, `data/processed/`)
- Model files (`*.h5`, `*.pkl`)
- Temporary files and logs
- IDE settings (can be customized)

This ensures you only commit code and configuration, not large binary files.
