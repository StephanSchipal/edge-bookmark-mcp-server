import sys
import os

print("=== Python Environment Check ===")
print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print("\n=== Python Path ===")
for p in sys.path:
    print(f" - {p}")

print("\n=== Import Tests ===")

def test_import(module_name):
    try:
        __import__(module_name)
        version = getattr(sys.modules[module_name], '__version__', 'unknown version')
        print(f"✅ {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name}: Unexpected error - {e}")
        return False

# Test standard library imports
test_import('os')
test_import('sys')
test_import('json')

# Test third-party imports
test_import('rapidfuzz')
test_import('aiofiles')
test_import('psutil')

# Test local imports
try:
    from search_engine import BookmarkSearchEngine
    print("✅ search_engine.BookmarkSearchEngine: Import successful")
    
    # Test creating an instance
    try:
        engine = BookmarkSearchEngine()
        print("✅ BookmarkSearchEngine: Instance created successfully")
    except Exception as e:
        print(f"❌ BookmarkSearchEngine: Failed to create instance - {e}")
    
except ImportError as e:
    print(f"❌ search_engine: {e}")
except Exception as e:
    print(f"⚠️ search_engine: Unexpected error - {e}")

print("\n=== Environment Check Complete ===")
