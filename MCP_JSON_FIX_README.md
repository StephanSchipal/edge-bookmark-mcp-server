# MCP JSON Communication Fix

## Problem
Edge-bookmarks MCP server was causing JSON parsing errors in Claude Desktop:
```
Unexpected non-whitespace character after JSON at position 4 (line 1 column 5)
```

## Root Cause
The server was logging emojis to `stdout` via `logging.StreamHandler(sys.stdout)`, which corrupted the JSON-RPC communication stream between Claude Desktop and the MCP server.

## Solution Applied
1. **Removed stdout handler** from logging configuration in `src/server.py`:
   ```python
   # OLD - PROBLEMATIC:
   handlers=[
       logging.StreamHandler(sys.stdout),  # This corrupts MCP JSON stream
       logging.FileHandler('edge-bookmark-server.log', encoding='utf-8')
   ]
   
   # NEW - FIXED:
   handlers=[
       # logging.StreamHandler(sys.stdout),  # REMOVED: Causes JSON corruption
       logging.FileHandler('edge-bookmark-server.log', encoding='utf-8')
   ]
   ```

2. **Silenced mock Context class** to prevent print statements when FastMCP unavailable

## Files Changed
- `src/server.py` (lines ~47 and ~30)

## Testing
The server will now log only to the file `edge-bookmark-server.log` and should not interfere with MCP JSON-RPC communication.

## Note
All server logs are still captured in the log file. The emoji corruption was specifically caused by stdout interference with the MCP protocol.
