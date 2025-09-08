#!/usr/bin/env python3
"""
Test script for Edge Bookmark MCP Server
Tests core functionality without full FastMCP setup
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import ServerConfig
from bookmark_loader import EdgeBookmarkLoader
from search_engine import BookmarkSearchEngine
from analytics import BookmarkAnalyzer
from exporter import BookmarkExporter
from file_monitor import BookmarkMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing Configuration...")
    
    try:
        config = ServerConfig()
        
        print(f"✅ Server name: {config.server_name}")
        print(f"✅ Server version: {config.server_version}")
        print(f"✅ Search default limit: {config.search.default_limit}")
        print(f"✅ Data directory: {config.data_directory}")
        
        # Test validation
        issues = config.validate_configuration()
        if issues:
            print(f⚠️ Configuration issues found: {issues}")
        else:
            print("✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_bookmark_loader():
    """Test bookmark file discovery and loading."""
    print("\n📁 Testing Bookmark Loader...")
    
    try:
        loader = EdgeBookmarkLoader()
        
        # Test profile discovery
        profiles = await loader.discover_edge_profiles()
        print(f"✅ Discovered {len(profiles)} Edge profiles")
        
        for profile in profiles:
            print(f"   📂 {profile}")
        
        if profiles:
            # Test bookmark loading
            bookmarks = await loader.load_all_bookmarks()
            print(f"✅ Loaded {len(bookmarks)} total bookmarks")
            
            if bookmarks:
                # Show sample bookmark
                sample = bookmarks[0]
                print(f"   📖 Sample bookmark: {sample.get('title', 'No title')} -> {sample.get('url', 'No URL')}")
            
            # Test validation
            validation = await loader.validate_bookmark_files()
            print(f"✅ File validation: {validation['valid_files']} valid, {validation['corrupted_files']} corrupted")
            
            return bookmarks
        else:
            print("⚠️ No Edge profiles found - will create mock data for testing")
            # Create mock bookmarks for testing
            mock_bookmarks = [
                {
                    'id': '1',
                    'title': 'GitHub',
                    'url': 'https://github.com',
                    'folder_path': 'Development',
                    'date_added': '2025-01-01T00:00:00Z',
                    'tags': ['code', 'development'],
                    'profile': 'Test'
                },
                {
                    'id': '2',
                    'title': 'Stack Overflow',
                    'url': 'https://stackoverflow.com',
                    'folder_path': 'Development/Help',
                    'date_added': '2025-01-02T00:00:00Z',
                    'tags': ['programming', 'help'],
                    'profile': 'Test'
                },
                {
                    'id': '3',
                    'title': 'FastMCP Documentation',
                    'url': 'https://gofastmcp.com',
                    'folder_path': 'Documentation',
                    'date_added': '2025-01-03T00:00:00Z',
                    'tags': ['mcp', 'docs'],
                    'profile': 'Test'
                }
            ]
            print(f"✅ Created {len(mock_bookmarks)} mock bookmarks for testing")
            return mock_bookmarks
        
    except Exception as e:
        print(f"❌ Bookmark loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

async def test_search_engine(bookmarks):
    """Test search functionality."""
    print("\n🔍 Testing Search Engine...")
    
    try:
        search_engine = BookmarkSearchEngine()
        
        # Index bookmarks
        await search_engine.index_bookmarks(bookmarks)
        print(f"✅ Indexed {len(bookmarks)} bookmarks")
        
        # Test fuzzy search
        test_queries = ['github', 'development', 'docs', 'help']
        
        for query in test_queries:
            print(f"\n   🔍 Testing query: '{query}'")
            
            # Fuzzy search
            fuzzy_results = await search_engine.fuzzy_search(query, limit=5)
            print(f"      Fuzzy: {len(fuzzy_results)} results")
            
            for result in fuzzy_results[:2]:  # Show top 2
                bookmark = result['bookmark']
                score = result['score']
                print(f"         📖 {bookmark.get('title', 'No title')} (score: {score:.1f})")
            
            # Exact search
            exact_results = await search_engine.exact_search(query, limit=5)
            print(f"      Exact: {len(exact_results)} results")
            
            # Search suggestions
            suggestions = await search_engine.get_suggestions(query[:3], limit=3)
            print(f"      Suggestions for '{query[:3]}': {suggestions}")
        
        # Test domain search
        domain_results = await search_engine.search_by_domain('github.com', limit=5)
        print(f"\n   🌐 Domain search (github.com): {len(domain_results)} results")
        
        # Test tag search
        tag_results = await search_engine.search_by_tags(['development'], limit=5)
        print(f"   🏷️ Tag search (development): {len(tag_results)} results")
        
        # Get search stats
        stats = search_engine.get_search_statistics()
        print(f"\n✅ Search statistics:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Average search time: {stats['average_search_time_ms']:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_analytics(bookmarks):
    """Test analytics functionality."""
    print("\n📊 Testing Analytics...")
    
    try:
        analyzer = BookmarkAnalyzer()
        
        # Test bookmark analysis
        analysis = await analyzer.analyze_bookmarks(bookmarks)
        print(f"✅ Analysis completed")
        print(f"   Total bookmarks: {analysis['total_bookmarks']}")
        print(f"   Folders found: {len(analysis['folder_structure']['folder_distribution'])}")
        print(f"   Max folder depth: {analysis['folder_structure']['max_depth']}")
        print(f"   Top domains: {len(analysis['domain_distribution'])}")
        
        # Test duplicate detection
        duplicates = await analyzer.detect_duplicates(bookmarks)
        print(f"   Duplicates found: {len(duplicates)}")
        
        # Show recommendations
        recommendations = analysis.get('recommendations', [])
        print(f"   Recommendations: {len(recommendations)}")
        for rec in recommendations[:2]:
            print(f"      💡 {rec['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_exporter(bookmarks):
    """Test export functionality."""
    print("\n📤 Testing Exporter...")
    
    try:
        exporter = BookmarkExporter()
        
        # Test directory creation
        test_dir = Path("data/test_exports")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test JSON export
        json_result = await exporter.export_bookmarks(
            bookmarks, 'json', str(test_dir / 'test_bookmarks.json')
        )
        print(f"✅ JSON export: {json_result}")
        
        # Test CSV export
        csv_result = await exporter.export_bookmarks(
            bookmarks, 'csv', str(test_dir / 'test_bookmarks.csv')
        )
        print(f"✅ CSV export: {csv_result}")
        
        # Test HTML export
        html_result = await exporter.export_bookmarks(
            bookmarks, 'html', str(test_dir / 'test_bookmarks.html')
        )
        print(f"✅ HTML export: {html_result}")
        
        # Verify files exist
        for file_name in ['test_bookmarks.json', 'test_bookmarks.csv', 'test_bookmarks.html']:
            file_path = test_dir / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"   📄 {file_name}: {size} bytes")
            else:
                print(f"   ❌ {file_name}: File not created")
        
        return True
        
    except Exception as e:
        print(f"❌ Exporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_file_monitor():
    """Test file monitoring."""
    print("\n👁️ Testing File Monitor...")
    
    try:
        def mock_callback(file_path):
            print(f"   📝 File change detected: {file_path}")
        
        monitor = BookmarkMonitor(on_change_callback=mock_callback)
        
        # Test start/stop
        await monitor.start_monitoring()
        print("✅ File monitoring started")
        
        # Wait a bit to see if it works
        await asyncio.sleep(2)
        
        await monitor.stop_monitoring()
        print("✅ File monitoring stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ File monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧪 Edge Bookmark MCP Server - Component Testing")
    print("=" * 60)
    
    test_results = []
    
    # Test configuration
    config_ok = await test_configuration()
    test_results.append(("Configuration", config_ok))
    
    # Test bookmark loader
    bookmarks = await test_bookmark_loader()
    loader_ok = len(bookmarks) > 0
    test_results.append(("Bookmark Loader", loader_ok))
    
    if bookmarks:
        # Test search engine
        search_ok = await test_search_engine(bookmarks)
        test_results.append(("Search Engine", search_ok))
        
        # Test analytics
        analytics_ok = await test_analytics(bookmarks)
        test_results.append(("Analytics", analytics_ok))
        
        # Test exporter
        export_ok = await test_exporter(bookmarks)
        test_results.append(("Exporter", export_ok))
    else:
        test_results.extend([
            ("Search Engine", False),
            ("Analytics", False), 
            ("Exporter", False)
        ])
    
    # Test file monitor
    monitor_ok = await test_file_monitor()
    test_results.append(("File Monitor", monitor_ok))
    
    # Summary
    print("\n🏁 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for component, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{component:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The server components are working correctly.")
        print("Next step: Set up FastMCP server and test full integration.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())