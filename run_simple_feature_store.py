#!/usr/bin/env python
"""
Wrapper script for simple feature store example
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from examples.simple_feature_store import main
    main()
