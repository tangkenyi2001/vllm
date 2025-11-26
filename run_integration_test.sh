#!/bin/bash
# Script to run the full integration test for collective_rpc

echo "========================================================================"
echo "collective_rpc Integration Test Runner"
echo "========================================================================"
echo ""

# Check if torch is available
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠  WARNING: PyTorch (torch) is not installed or not in PATH"
    echo ""
    echo "This test requires PyTorch to run. Please either:"
    echo ""
    echo "1. Activate your conda environment:"
    echo "   conda activate vllm"
    echo ""
    echo "2. Or activate your virtual environment:"
    echo "   source /path/to/your/venv/bin/activate"
    echo ""
    echo "3. Or install PyTorch:"
    echo "   pip install torch"
    echo ""
    echo "Then run this script again or run directly:"
    echo "   python test_full_integration.py"
    echo ""
    exit 1
fi

echo "✓ PyTorch is available"
echo ""

# Run the test
python test_full_integration.py

exit $?
