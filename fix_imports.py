# Fix relative imports in all src files
import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix import statements to use relative imports"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix patterns
    fixes = [
        (r'^from config import', 'from .config import'),
        (r'^from bookmark_loader import', 'from .bookmark_loader import'),
        (r'^from search_engine import', 'from .search_engine import'),
        (r'^from analytics import', 'from .analytics import'),
        (r'^from exporter import', 'from .exporter import'),
        (r'^from file_monitor import', 'from .file_monitor import'),
        (r'^import config$', 'from . import config'),
    ]
    
    modified = False
    for pattern, replacement in fixes:
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            modified = True
            print(f"Fixed import in {filepath}: {pattern} -> {replacement}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Fix all Python files in src/
src_dir = Path('src')
python_files = list(src_dir.glob('*.py'))

print("Fixing imports in Python files...")
for py_file in python_files:
    if py_file.name not in ['__init__.py', '__main__.py']:
        if fix_imports_in_file(py_file):
            print(f"✅ Fixed imports in {py_file}")
        else:
            print(f"ℹ️  No import fixes needed in {py_file}")
