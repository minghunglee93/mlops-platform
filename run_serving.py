#!/usr/bin/env python
"""
Simple wrapper script to run serving API with proper path setup
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now run the serving API
if __name__ == "__main__":
    import uvicorn
    from serving.api import app
    from config import settings
    
    uvicorn.run(
        app,
        host=settings.SERVING_HOST,
        port=settings.SERVING_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
