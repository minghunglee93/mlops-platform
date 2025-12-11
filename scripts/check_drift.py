#!/usr/bin/env python
"""Check if model needs retraining due to drift."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import requests

def check_drift(model_name: str, api_url: str = "http://localhost:8000") -> bool:
    """Check if retraining needed."""
    try:
        response = requests.get(f"{api_url}/retraining/{model_name}/check", timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get('needs_retraining', False)
        return False
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="classifier_model")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    
    needs_retraining = check_drift(args.model, args.api_url)
    print("true" if needs_retraining else "false")
