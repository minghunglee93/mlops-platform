#!/usr/bin/env python
"""
Wrapper script for A/B testing example
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from examples.ab_testing_example import main
    main()