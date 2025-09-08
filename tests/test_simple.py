#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("✅ Python environment is working")

# Test imports
try:
    import rapidfuzz
    print(f"✅ RapidFuzz {rapidfuzz.__version__} is available")
    
    from search_engine import BookmarkSearchEngine
    print("✅ BookmarkSearchEngine imported successfully")
    
    # Initialize search engine
    search_engine = BookmarkSearchEngine()
    print("✅ Search engine initialized")
    
    # Test basic search using fuzzy_search
    import asyncio
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(search_engine.fuzzy_search("test", limit=3))
    print(f"✅ Basic search completed. Found {len(results)} results")
    if results:
        print("Sample result:", results[0].get('title', 'No title'))
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    raise
