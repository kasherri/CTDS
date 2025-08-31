#!/bin/bash
set -e  # Exit on any error

# CTDS Setup Script
# This script sets up the Cell-Type Dynamical Systems (CTDS) package
# Author: Njeri Njoroge <njerinjoroge9@gmail.com>

echo "🧠 Setting up Cell-Type Dynamical Systems (CTDS)..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Error: Python $python_version is too old. Please install Python $required_version or later."
    exit 1
fi

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
    VENV_PATH="$VIRTUAL_ENV"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    # Create virtual environment if it doesn't exist
    VENV_PATH=".venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    else
        echo "✅ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    echo "🔌 Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    PYTHON_CMD="$VENV_PATH/bin/python"
    PIP_CMD="$VENV_PATH/bin/pip"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install build dependencies
echo "🔧 Installing build dependencies..."
$PIP_CMD install --upgrade setuptools wheel build

# Install CTDS package in development mode with all dependencies
echo "📥 Installing CTDS package..."
$PIP_CMD install -e .

# Install development dependencies
echo "🛠️  Installing development dependencies..."
$PIP_CMD install -e ".[dev]"

# Install documentation dependencies
echo "📚 Installing documentation dependencies..."
$PIP_CMD install -e ".[docs]"

# Install example dependencies
echo "📊 Installing example dependencies..."
$PIP_CMD install -e ".[examples]"

# Verify installation
echo "🧪 Verifying installation..."
if $PYTHON_CMD -c "import ctds; print(f'✅ CTDS {ctds.__version__ if hasattr(ctds, \"__version__\") else \"0.1.0\"} imported successfully')"; then
    echo "✅ CTDS package installed successfully!"
else
    echo "❌ Error: CTDS package failed to import"
    exit 1
fi

# Run basic tests to verify everything works
echo "🧪 Running basic tests..."
if $PYTHON_CMD -m pytest tests/test_models.py::TestCTDSModel::test_model_creation -v --tb=short; then
    echo "✅ Basic tests passed!"
else
    echo "⚠️  Warning: Some basic tests failed. Check the installation."
fi

# Check JAX installation
echo "🔍 Checking JAX installation..."
if $PYTHON_CMD -c "import jax; print(f'✅ JAX {jax.__version__} with backend: {jax.default_backend()}')"; then
    echo "✅ JAX is working properly!"
else
    echo "❌ Error: JAX installation failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Quick start:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run demo: python quickstart.py"
echo "  3. Run tests: python -m pytest tests/"
echo "  4. View examples: jupyter notebook"
echo "  5. Build docs: mkdocs serve"
echo ""
echo "📖 See README.MD for usage examples and documentation."
echo ""

# Provide environment activation instructions
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "💡 To activate the environment in the future, run:"
    echo "   source .venv/bin/activate"
    echo ""
fi
