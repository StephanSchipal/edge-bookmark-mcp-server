#!/usr/bin/env python3
"""
Windsurf Component Testing Script
Tests the Edge Bookmark MCP Server components
"""

import asyncio
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
logger = logging.getLogger(__name__)

async def test_imports():
    """Test if all modules can be imported."""
    print("\n🔍 Testing Module Imports...")
    
    try:
        from config import ServerConfig
        print("✅ config module imported successfully")
        
        from bookmark_loader import EdgeBookmarkLoader  
        print("✅ bookmark_loader module imported successfully")
        
        from search_engine import BookmarkSearchEngine
        print("✅ search_engine module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_configuration():
    """Test configuration system."""
    print("\n🔧 Testing Configuration...")
    
    try:
        from config import config
        
        print(f"✅ Server: {config.server_name} v{config.server_version}")
        print(f"✅ Search limit: {config.search.default_limit}")
        print(f"✅ Data directory: {config.data_directory}")
        
        # Test validation
        issues = config.validate_configuration()
        if issues:
            print(f"⚠️ Configuration issues: {issues}")
        else:
            print("✅ Configuration validation passed")
        
        # Test serialization
        config_dict = config.to_dict()
        print(f"✅ Configuration serializable: {len(config_dict)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_bookmark_loader():
    """Test bookmark loading functionality."""
    print("\n📁 Testing Bookmark Loader...")
    
    try:
        from bookmark_loader import EdgeBookmarkLoader
        
        loader = EdgeBookmarkLoader()
        print("✅ BookmarkLoader created")
        
        # Test profile discovery
        profiles = await loader.discover_edge_profiles()
        print(f"✅ Profile discovery: {len(profiles)} profiles found")
        
        for profile in profiles:
            print(f"   📂 {profile}")
        
        # Test bookmark loading
        bookmarks = await loader.load_all_bookmarks()
        print(f"✅ Bookmark loading: {len(bookmarks)} bookmarks loaded")
        
        if bookmarks:
            # Show sample bookmarks
            print("\n📖 Sample bookmarks:")
            for i, bookmark in enumerate(bookmarks[:3]):
                title = bookmark.get('title', 'No title')[:50]
                url = bookmark.get('url', 'No URL')[:60]
                folder = bookmark.get('folder_path', 'No folder')
                print(f"   {i+1}. {title} -> {url}")
                print(f"      📁 {folder}")
        
        # Test validation
        validation = await loader.validate_bookmark_files()
        print(f"✅ File validation: {validation}")
        
        return bookmarks
        
    except Exception as e:
        print(f"❌ Bookmark loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

async def test_search_engine(bookmarks):
    """Test search engine functionality."""
    print("\n🔍 Testing Search Engine...")
    
    try:
        from search_engine import BookmarkSearchEngine
        
        search_engine = BookmarkSearchEngine()
        print("✅ SearchEngine created")
        
        # Test indexing
        await search_engine.index_bookmarks(bookmarks)
        print(f"✅ Indexing completed for {len(bookmarks)} bookmarks")
        
        # Test searches
        test_queries = ['github', 'python', 'development', 'docs', 'ai']
        
        for query in test_queries:
            print(f"\n   🔍 Testing query: '{query}'")
            
            # Fuzzy search
            fuzzy_results = await search_engine.fuzzy_search(query, limit=3)
            print(f"      Fuzzy search: {len(fuzzy_results)} results")
            
            for i, result in enumerate(fuzzy_results):
                bookmark = result['bookmark']
                score = result['score']
                title = bookmark.get('title', 'No title')[:40]
                print(f"         {i+1}. {title} (score: {score:.1f})")
            
            # Test suggestions
            if len(query) >= 3:
                suggestions = await search_engine.get_suggestions(query[:3], limit=3)
                print(f"      Suggestions for '{query[:3]}': {suggestions}")
        
        # Test domain search
        domain_results = await search_engine.search_by_domain('github.com', limit=2)
        print(f"\n   🌐 Domain search (github.com): {len(domain_results)} results")
        
        # Test tag search
        tag_results = await search_engine.search_by_tags(['development'], limit=2)
        print(f"   🏷️ Tag search (development): {len(tag_results)} results")
        
        # Get statistics
        stats = search_engine.get_search_statistics()
        print(f"\n✅ Search statistics:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Average time: {stats['average_search_time_ms']:.2f}ms")
        print(f"   RapidFuzz: {stats['rapidfuzz_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_edge_detection():
    """Test Edge installation detection."""
    print("\n🌐 Testing Edge Detection...")
    
    import os
    
    # Possible Edge paths in Windsurf environment
    edge_paths = [
        # Windows paths
        Path.home() / "AppData/Local/Microsoft/Edge/User Data",
        # WSL paths
        Path(f"/mnt/c/Users/{os.environ.get('USER', 'user')}/AppData/Local/Microsoft/Edge/User Data"),
        Path(f"/c/Users/{os.environ.get('USER', 'user')}/AppData/Local/Microsoft/Edge/User Data"),
        # Common system paths
        Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application"),
        Path("/c/Program Files (x86)/Microsoft/Edge/Application")
    ]
    
    edge_found = False
    bookmarks_found = False
    
    for edge_path in edge_paths:
        if edge_path.exists():
            print(f"✅ Edge directory found: {edge_path}")
            edge_found = True
            
            # Check for bookmarks
            bookmark_locations = [
                edge_path / "Default/Bookmarks",
                edge_path / "Profile 1/Bookmarks"
            ]
            
            for bookmark_path in bookmark_locations:
                if bookmark_path.exists():
                    size = bookmark_path.stat().st_size
                    print(f"   📖 Bookmarks found: {bookmark_path} ({size} bytes)")
                    bookmarks_found = True
            break
    
    if not edge_found:
        print("⚠️ Edge installation not detected in standard locations")
        print("   This is normal in WSL/Windsurf environments")
    
    if not bookmarks_found:
        print("📋 No bookmark files found - tests will use mock data")
    
    return edge_found, bookmarks_found

async def test_file_operations():
    """Test file operations."""
    print("\n📄 Testing File Operations...")
    
    try:
        # Test data directory creation
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        print(f"✅ Data directory: {data_dir.absolute()}")
        
        # Test subdirectories
        subdirs = ['exports', 'backups', 'cache']
        for subdir in subdirs:
            (data_dir / subdir).mkdir(exist_ok=True)
            print(f"✅ Created: data/{subdir}")
        
        # Test file writing
        test_file = data_dir / "test.txt"
        test_file.write_text("Test file content")
        print(f"✅ File write test: {test_file}")
        
        # Test file reading
        content = test_file.read_text()
        print(f"✅ File read test: {len(content)} characters")
        
        # Cleanup
        test_file.unlink()
        print("✅ File cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False

async def test_dependencies():
    """Test optional dependencies."""
    print("\n📦 Testing Dependencies...")
    
    # Test RapidFuzz
    try:
        import rapidfuzz
        print(f"✅ RapidFuzz available: {rapidfuzz.__version__}")
    except ImportError:
        print("⚠️ RapidFuzz not available - using basic string matching")
    
    # Test other dependencies
    deps = ['asyncio', 'json', 'pathlib', 're', 'logging']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")
    
    # Test optional dependencies
    optional_deps = ['pandas', 'aiofiles', 'psutil']
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available (optional)")
        except ImportError:
            print(f"⚠️ {dep} not available (optional)")
    
    return True

async def main():
    """Main test runner."""
    print("🧪 Edge Bookmark MCP Server - Windsurf Component Testing")
    print("=" * 65)
    
    # Test results tracking
    tests = []
    
    # Run tests
    tests.append(("Module Imports", await test_imports()))
    tests.append(("Configuration", await test_configuration()))
    tests.append(("Dependencies", await test_dependencies()))
    tests.append(("Edge Detection", (await test_edge_detection())[0]))
    tests.append(("File Operations", await test_file_operations()))
    
    # Bookmark tests
    bookmarks = await test_bookmark_loader()
    tests.append(("Bookmark Loader", len(bookmarks) > 0))
    
    if bookmarks:
        tests.append(("Search Engine", await test_search_engine(bookmarks)))
    else:
        tests.append(("Search Engine", False))
    
    # Summary
    print("\n🏁 Test Results Summary")
    print("=" * 65)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if passed == total:
        print("🎉 All tests passed! Components are working correctly.")
        print("Next steps:")
        print("  1. Install missing dependencies: pip install rapidfuzz aiofiles psutil")
        print("  2. Create FastMCP server integration")
        print("  3. Test with Windsurf MCP configuration")
    else:
        print("⚠️ Some tests failed. Common solutions:")
        print("  1. Ensure you're in the correct directory")
        print("  2. Check Python path and imports")
        print("  3. Install missing dependencies")
        
        if any("Search Engine" in test[0] and not test[1] for test in tests):
            print("  4. Search issues are often due to missing RapidFuzz")
    
    # Environment info
    print(f"\n📊 Environment Info:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Dir: {Path.cwd()}")
    print(f"  Script Dir: {Path(__file__).parent}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)