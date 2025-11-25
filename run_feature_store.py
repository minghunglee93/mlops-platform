#!/usr/bin/env python
"""
Simple wrapper script to run feature store example with proper path setup
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the example
if __name__ == "__main__":
    from examples.feature_store_example import main
    main()