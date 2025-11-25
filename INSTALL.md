# Installation Guide

## Method 1: Development Installation (Recommended)

This method installs the package in editable mode, allowing you to modify the code.

```bash
# 1. Clone the repository
git clone <repo-url>
cd mlops-platform

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install in editable mode
pip install -e .

# This will install all dependencies and make the modules importable
```

## Method 2: Package Installation

```bash
# Install directly from the directory
pip install .

# Or install from git repository
pip install git+https://github.com/yourusername/mlops-platform.git
```

## Method 3: Manual Setup (If setup.py doesn't work)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies only
pip install -r requirements.txt

# 3. Add the project to PYTHONPATH
# On Linux/Mac:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# On Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%

# 4. Run from the root directory
python examples/train_example.py
```

## Verifying Installation

After installation, verify everything works:

```bash
# Check if packages are importable
python -c "from training.pipeline import TrainingPipeline; print('✓ Training module OK')"
python -c "from registry.model_registry import ModelRegistry; print('✓ Registry module OK')"
python -c "from config import settings; print('✓ Config module OK')"

# Run the example
python examples/train_example.py
```

## Troubleshooting

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'training'`

**Solution 1** (Recommended):
```bash
# Install package in editable mode
pip install -e .
```

**Solution 2**:
```bash
# Run from project root and use Python's module syntax
python -m examples.train_example
```

**Solution 3**:
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/train_example.py
```

### Import Errors with Dependencies

**Problem**: Dependency conflicts or version mismatches

**Solution**:
```bash
# Create fresh virtual environment
python -m venv venv_fresh
source venv_fresh/bin/activate
pip install --upgrade pip
pip install -e .
```

### MLflow Database Locked

**Problem**: `database is locked` error

**Solution**:
```bash
# Stop any running MLflow servers
pkill -f "mlflow"

# Delete lock file (if exists)
rm mlflow.db-shm mlflow.db-wal

# Restart
python examples/train_example.py
```

## Directory Structure After Installation

```
mlops-platform/
├── __init__.py              # Makes it a package
├── config.py
├── training/
│   ├── __init__.py         # Important!
│   └── pipeline.py
├── registry/
│   ├── __init__.py         # Important!
│   └── model_registry.py
├── serving/
│   ├── __init__.py         # Important!
│   └── api.py
├── examples/
│   ├── __init__.py         # Important!
│   └── train_example.py
├── setup.py                 # Package setup
├── requirements.txt
└── venv/                    # Virtual environment
```

## Quick Start Commands

After successful installation:

```bash
# Train models
python examples/train_example.py

# Or use the installed command (if using Method 1 or 2)
mlops-train

# Start MLflow UI
mlflow ui

# Start serving API
python serving/api.py
# Or
mlops-serve
```

## Next Steps

1. ✅ Verify installation
2. ✅ Run example training
3. ✅ View experiments in MLflow UI
4. ✅ Start the serving API
5. ✅ Make test predictions

See [README.md](README.md) for detailed usage instructions.
