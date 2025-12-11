#!/usr/bin/env python
"""Promote model to production stage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from registry.model_registry import ModelRegistry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def promote_model(model_name: str, stage: str = "Production") -> bool:
    """Promote model to specified stage."""
    registry = ModelRegistry()
    
    versions = registry.get_model_versions(model_name)
    if not versions:
        logger.error(f"No versions found for {model_name}")
        return False
    
    latest = versions[0]
    version = latest['version']
    
    success = registry.promote_model(model_name, version, stage)
    
    if success:
        logger.info(f"✓ Promoted {model_name} v{version} to {stage}")
    else:
        logger.error(f"✗ Failed to promote {model_name}")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()
    
    success = promote_model(args.model, args.stage)
    sys.exit(0 if success else 1)
