# Quick Start Guide - MLOps Platform

## 🚀 Get Started in 3 Minutes

### Step 1: Setup (1 minute)

```bash
# Clone and navigate
cd mlops-platform

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies (includes Feast for feature store)
pip install -r requirements.txt
```

### Step 2: Train a Model (1 minute)

```bash
# Use the simple wrapper script
python run_training.py
```

**That's it!** Your model is trained and tracked in MLflow.

### Step 2b: Try Feature Store (NEW!)

```bash
# Simple, reliable example
python run_simple_feature_store.py

# Full example (more complex)
python run_feature_store.py
```

**Features engineered and served!** Start with the simple example to understand basics.

### Step 3: View Results (30 seconds)

```bash
# Start MLflow UI (in a new terminal)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Open browser to: http://localhost:5000
```

### Step 4: Serve the Model (30 seconds)

```bash
# Start API server
python run_serving.py

# API docs at: http://localhost:8000/docs
```

### Step 5: Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]],
    "model_name": "classifier_model"
  }'
```

---

## 📝 Three Ways to Run

### Method 1: Wrapper Scripts (Recommended for Beginners)

```bash
python run_training.py   # Train models
python run_serving.py    # Start API
```

✅ Pros: Just works, no setup needed
❌ Cons: Only for provided scripts

### Method 2: Install as Package (Recommended for Development)

```bash
# One-time setup
pip install -e .

# Now run anything
python examples/train_example.py
python serving/api.py
python -c "from training import TrainingPipeline"
```

✅ Pros: Professional setup, import from anywhere
❌ Cons: One extra install step

### Method 3: PYTHONPATH (Advanced)

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/train_example.py
```

✅ Pros: No installation needed
❌ Cons: Need to set PYTHONPATH every time

---

## 🔧 Common Commands

```bash
# Training
python run_training.py

# Feature Store (NEW!)
python run_feature_store.py

# View experiments
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Explore features (NEW!)
cd feature_repo && feast feature-views list

# Start API
python run_serving.py

# Test API
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[...]], "model_name": "classifier_model"}'
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'config'"

**Solution:**
```bash
# Use the wrapper script instead
python run_training.py  # ✅ Instead of: python examples/train_example.py
```

Or install as package:
```bash
pip install -e .
```

### "Address already in use"

```bash
# Find and kill the process
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Or use a different port
python run_serving.py  # Edit the file to change port
```

### "MLflow database is locked"

```bash
pkill -f mlflow  # Kill all MLflow processes
rm -f mlflow.db-shm mlflow.db-wal  # Remove lock files
```

---

## 📁 Project Structure

```
mlops-platform/
├── run_training.py      ← Run this to train models
├── run_serving.py       ← Run this to start API
├── config.py            ← Settings
├── training/            
│   └── pipeline.py      ← Training logic
├── registry/            
│   └── model_registry.py ← Model versioning
├── serving/             
│   └── api.py           ← REST API
└── examples/            
    └── train_example.py ← Example training script
```

---

## 💡 Tips

1. **Always activate venv first**: `source venv/bin/activate`
2. **Use wrapper scripts**: They handle imports automatically
3. **Check logs**: Look in `logs/` for errors
4. **Start fresh**: Delete `mlflow.db` and `models/` to reset

---

## 🎯 Next Steps

1. ✅ **Customize training**: Edit `examples/train_example.py`
2. ✅ **Add your data**: Replace synthetic data with real data
3. ✅ **Try different models**: Add more models to compare
4. ✅ **Monitor production**: Check Prometheus metrics at `/metrics`
5. ✅ **Deploy**: Use Docker (see `Dockerfile`)

---

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Installation help: [INSTALL.md](INSTALL.md)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Need help?** Check the troubleshooting guide or open an issue!
