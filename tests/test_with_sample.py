#!/usr/bin/env python3
"""
Test script with sample bookmarks
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
import logging
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
    
    # Create some sample bookmarks
    sample_bookmarks = [
        {
            "name": "Test Bookmark 1",
            "url": "https://example1.com",
            "date_added": "2023-01-01",
            "tags": ["test", "example"]
        },
        {
            "name": "Another Test Bookmark",
            "url": "https://example2.com",
            "date_added": "2023-01-02",
            "tags": ["test", "demo"]
        },
        {
            "name": "Python Documentation",
            "url": "https://docs.python.org",
            "date_added": "2023-01-03",
            "tags": ["python", "documentation"]
        }
    ]
    
    # Index the sample bookmarks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(search_engine.index_bookmarks(sample_bookmarks))
    print("✅ Indexed sample bookmarks")
    
    # Test fuzzy search
    results = loop.run_until_complete(search_engine.fuzzy_search("test", limit=2))
    print(f"✅ Fuzzy search found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('name')} - {result.get('url')}")
    
    # Test exact search
    results = loop.run_until_complete(search_engine.exact_search("python", limit=1))
    print(f"✅ Exact search found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('name')} - {result.get('url')}")
    
    # Test search by tags
    results = loop.run_until_complete(search_engine.search_by_tags(["example"], limit=2))
    print(f"✅ Tag search found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('name')} - {result.get('url')}")
    
    print("✅ All tests completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    raise
