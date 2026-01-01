#!/bin/bash
# Script to fix scipy/numpy compatibility issue
# Run this within the conda environment

echo "Fixing scipy/numpy compatibility issue..."
echo "Current versions:"
/Users/christos/anaconda3/envs/friederici_llm/bin/python -c "import numpy; import scipy; print(f'NumPy: {numpy.__version__}'); print(f'SciPy: {scipy.__version__}')"

echo ""
echo "Updating NumPy and SciPy to compatible versions..."
/Users/christos/anaconda3/envs/friederici_llm/bin/pip install --upgrade "numpy>=1.26.0,<1.27" "scipy>=1.11.0,<1.13.0" --force-reinstall

echo ""
echo "Testing import..."
/Users/christos/anaconda3/envs/friederici_llm/bin/python -c "import scipy.stats; print('✓ scipy.stats imports successfully!')" && echo "Success!" || echo "Still having issues - you may need to reinstall the environment"

