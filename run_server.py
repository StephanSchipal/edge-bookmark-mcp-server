#!/usr/bin/env python3
"""
Edge Bookmark MCP Server - Direct Runner
Simple entry point that avoids module import issues
"""
import sys
import os
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run
if __name__ == "__main__":
    try:
        from server import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"📁 Current directory: {Path.cwd()}")
        print(f"📁 Source directory: {src_dir}")
        print(f"📁 Source exists: {src_dir.exists()}")
        if src_dir.exists():
            print(f"📄 Files in src: {list(src_dir.glob('*.py'))}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
