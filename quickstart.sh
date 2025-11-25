#!/bin/bash
# Quick start script for MLOps Platform

set -e  # Exit on error

echo "=========================================="
echo "MLOps Platform - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

# Install package in editable mode
echo "Installing MLOps Platform..."
pip install -e . > /dev/null 2>&1
echo "✓ Package installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models logs
echo "✓ Directories created"
echo ""

# Copy environment template
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
else
    echo "⚠ .env file already exists. Skipping..."
fi
echo ""

# Verify installation
echo "Verifying installation..."
python -c "from training.pipeline import TrainingPipeline" 2>/dev/null && echo "✓ Training module OK" || echo "✗ Training module FAILED"
python -c "from registry.model_registry import ModelRegistry" 2>/dev/null && echo "✓ Registry module OK" || echo "✗ Registry module FAILED"
python -c "from config import settings" 2>/dev/null && echo "✓ Config module OK" || echo "✗ Config module FAILED"
echo ""

echo "=========================================="
echo "Installation Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Train your first model:"
echo "     python run_training.py"
echo ""
echo "  3. View experiments:"
echo "     mlflow ui"
echo ""
echo "  4. Start serving API:"
echo "     python run_serving.py"
echo ""
echo "💡 Tip: Use run_training.py and run_serving.py"
echo "   (they handle imports automatically!)"
echo ""
echo "For more information, see QUICKSTART_GUIDE.md"
echo ""