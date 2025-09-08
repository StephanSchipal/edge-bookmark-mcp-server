#!/usr/bin/env python3
"""
Edge Bookmark MCP Server - Module Entry Point
Handles both direct execution and module import scenarios
"""
import sys
import os
from pathlib import Path

# Ensure we can import from the src directory
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
src_dir = current_dir

# Add both current and parent to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Now import with absolute paths
try:
    from server import main
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

if __name__ == "__main__":
    main()
