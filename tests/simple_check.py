print("Python is working!")
print(f"Python version: {__import__('sys').version}")
print(f"Current directory: {__import__('os').getcwd()}")
try:
    import rapidfuzz
    print(f"RapidFuzz version: {rapidfuzz.__version__}")
except ImportError as e:
    print(f"RapidFuzz not available: {e}")
