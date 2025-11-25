# Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'config' or 'training'

**Symptoms:**
```
ModuleNotFoundError: No module named 'config'
ModuleNotFoundError: No module named 'training'
ModuleNotFoundError: No module named 'registry'
```

This is the most common issue! Here are multiple solutions:

**Solution 1** (Easiest - Use Wrapper Scripts):
```bash
# Instead of:
python examples/train_example.py  # ❌

# Use:
python run_training.py  # ✅
python run_serving.py   # ✅
```

**Solution 2** (Install as Package):
```bash
# From project root
pip install -e .

# Then you can run from anywhere
python examples/train_example.py
```

**Solution 3** (Set PYTHONPATH):
```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows

python examples/train_example.py
```

**Solution 4** (Run as Module):
```bash
# From project root
python -m examples.train_example
```

**Solution 5** (Manual Path Setup):
```python
# Add this at the top of your script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Then your imports
from config import settings
```

**Recommended**: Use Solution 1 (wrapper scripts) or Solution 2 (pip install -e .)

### 2. ModuleNotFoundError: No module named 'training' (but config works)

**Symptoms:**
```
ModuleNotFoundError: No module named 'training'
ModuleNotFoundError: No module named 'registry'
ModuleNotFoundError: No module named 'config'
```

**Solution 1** (Recommended):
```bash
# Install package in editable mode
pip install -e .
```

**Solution 2**:
```bash
# Run from project root directory
cd /path/to/mlops-platform
python examples/train_example.py
```

**Solution 3**:
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows

python examples/train_example.py
```

**Solution 4**:
```bash
# Use Python module syntax
python -m examples.train_example
```

### 2. Missing __init__.py Files

**Problem**: Directories not recognized as Python packages

**Solution**: Ensure these files exist:
```
mlops-platform/
├── __init__.py              ✓
├── training/__init__.py     ✓
├── registry/__init__.py     ✓
├── serving/__init__.py      ✓
└── examples/__init__.py     ✓
```

Create them with:
```bash
touch __init__.py
touch training/__init__.py
touch registry/__init__.py
touch serving/__init__.py
touch examples/__init__.py
```

### 3. MLflow Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solution**:
```bash
# Stop all MLflow processes
pkill -f mlflow

# Remove lock files
rm -f mlflow.db-shm mlflow.db-wal

# Restart
python examples/train_example.py
```

### 4. Dependency Version Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution 1** (Fresh install):
```bash
# Remove existing venv
rm -rf venv

# Create fresh environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

**Solution 2** (Force reinstall):
```bash
pip install --force-reinstall -e .
```

**Solution 3** (Individual packages):
```bash
# Install problematic packages separately
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install -e .
```

### 5. Permission Errors on Linux/Mac

**Symptoms:**
```
Permission denied: 'mlflow.db'
Permission denied: './models'
```

**Solution**:
```bash
# Fix permissions
chmod -R u+w .

# Or run with sudo (not recommended)
sudo python examples/train_example.py
```

### 6. Port Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows

# Or use different port
SERVING_PORT=8001 python serving/api.py
```

### 7. CUDA/GPU Issues

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.is_available() returns False
```

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or reduce batch size in config.py
TRAIN_BATCH_SIZE=16  # instead of 32
```

### 8. Import Errors with Specific Modules

**Problem**: Cannot import specific functions

**Solution**:
```bash
# Verify module structure
python -c "import training; print(training.__file__)"
python -c "from training import pipeline; print(pipeline.__file__)"
python -c "from training.pipeline import TrainingPipeline; print('OK')"

# If any fail, reinstall
pip uninstall mlops-platform
pip install -e .
```

### 9. MLflow UI Not Showing Experiments

**Problem**: MLflow UI is empty

**Solution**:
```bash
# Check database location
ls -la mlflow.db

# Specify tracking URI explicitly
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db
mlflow ui

# Or in Python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

### 10. FastAPI Startup Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'registry'
ImportError: cannot import name 'ModelRegistry'
```

**Solution**:
```bash
# Run from project root
cd /path/to/mlops-platform
python serving/api.py

# Or use absolute imports
export PYTHONPATH="$(pwd)"
python serving/api.py
```

## Verification Checklist

Run these commands to verify your setup:

```bash
# 1. Check Python version (should be 3.9+)
python --version

# 2. Check virtual environment is activated
which python  # Should point to venv/bin/python

# 3. Verify package installation
pip show mlops-platform

# 4. Test imports
python -c "from training.pipeline import TrainingPipeline; print('✓')"
python -c "from registry.model_registry import ModelRegistry; print('✓')"
python -c "from config import settings; print('✓')"

# 5. Check directory structure
ls -la __init__.py training/__init__.py registry/__init__.py

# 6. Verify MLflow
mlflow --version

# 7. Test basic training
python examples/train_example.py
```

## Getting Help

If you're still having issues:

1. **Check Logs**: Look in `logs/` directory for error details
2. **Verbose Mode**: Run with `LOG_LEVEL=DEBUG` in `.env`
3. **Clean Install**: Remove `venv/`, `mlflow.db`, and reinstall
4. **Update Packages**: `pip install --upgrade -r requirements.txt`
5. **Check Issues**: See GitHub issues for similar problems

## Debug Mode

Enable detailed logging:

```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOG_LEVEL=DEBUG
python examples/train_example.py
```

## Quick Reset

Start fresh:

```bash
# Remove all generated files
rm -rf venv/ mlflow.db* mlartifacts/ models/ __pycache__ */__pycache__

# Reinstall
python -m venv venv
source venv/bin/activate
pip install -e .

# Test
python examples/train_example.py
```