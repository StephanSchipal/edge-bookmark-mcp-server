# Edge Bookmark MCP Server

A comprehensive Model Context Protocol (MCP) server for Microsoft Edge bookmark management, built with FastMCP 2.12.2.

## ✨ Features

- **📚 Complete Bookmark Access**: Read, search, and manage Edge bookmarks
- **🔍 Advanced Search**: Search by title, URL, content, and tags
- **📁 Folder Management**: Navigate and organize bookmark folders
- **📊 Analytics**: Bookmark usage statistics and insights
- **📤 Export/Import**: Support for JSON, CSV, HTML, and XLSX formats
- **⚡ Real-time Monitoring**: Live bookmark file changes detection
- **🛡️ Safe Operations**: Read-only by default with optional write operations

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Microsoft Edge with bookmarks
- FastMCP 2.10.1+

### Installation

1. **Clone or download** this repository:
   `ash
   git clone https://github.com/stephanschipal/edge-bookmark-mcp.git
   cd edge-bookmark-mcp-server
   `

2. **Install dependencies**:
   `ash
   pip install fastmcp>=2.10.1 jinja2 pandas pydantic watchdog
   `

3. **Run the server**:
   `ash
   python run_server.py
   `

### Claude Desktop Configuration

Add to your Claude Desktop configuration (claude_desktop_config.json):

`json
{
  "mcpServers": {
    "edge-bookmarks": {
      "command": "python",
      "args": ["D:\\projects\\windsurf\\mcp\\edge-bookmark-mcp-server\\run_server.py"],
      "cwd": "D:\\projects\\windsurf\\mcp\\edge-bookmark-mcp-server"
    }
  }
}
`

## 🛠️ Available Tools

### Core Bookmark Operations
- list_bookmarks() - Get all bookmarks with folder structure
- search_bookmarks(query) - Search by title/URL/content
- get_bookmark_by_id(id) - Retrieve specific bookmark
- get_folder_contents(folder_id) - Browse bookmark folders

### Search & Analysis
- dvanced_search(filters) - Complex search with multiple criteria
- ind_duplicates() - Detect duplicate bookmarks
- nalyze_bookmarks() - Usage statistics and insights
- alidate_links() - Check for broken/dead links

### Export & Import
- export_bookmarks(format, options) - Export as JSON/CSV/HTML/XLSX
- import_bookmarks(file_path, format) - Import from various formats
- generate_report() - Create detailed bookmark reports

### Utility Functions
- get_server_status() - Server health and statistics
- efresh_bookmarks() - Reload bookmark data
- get_configuration() - View server settings

## 📁 Project Structure

`
edge-bookmark-mcp-server/
├── src/
│   ├── server.py           # Main FastMCP server
│   ├── bookmark_loader.py  # Edge bookmark file parsing
│   ├── search_engine.py    # Advanced search functionality
│   ├── analytics.py        # Bookmark analysis tools
│   ├── exporter.py         # Export/import capabilities
│   ├── file_monitor.py     # Real-time file monitoring
│   └── config.py           # Configuration management
├── tests/                  # Comprehensive test suite
├── data/                   # Sample and backup data
├── docs/                   # Additional documentation
├── examples/               # Usage examples
├── run_server.py           # Simple server launcher
├── pyproject.toml          # Python packaging
└── README.md               # This file
`

## ⚙️ Configuration

The server supports extensive configuration through src/config.py:

- **Bookmark file paths** (auto-detected for different Edge profiles)
- **Search indexing options**
- **Export templates and formats**
- **File monitoring settings**
- **Security and access controls**

## 🔒 Security

- **Read-only by default**: Safe bookmark access without modifications
- **Path validation**: Prevents access to unauthorized files
- **Error handling**: Graceful handling of file access issues
- **Logging**: Comprehensive audit trail of operations

## 📊 Usage Examples

### Basic Search
`python
# Search for bookmarks about "github"
results = search_bookmarks("github")
`

### Export Bookmarks
`python
# Export all bookmarks as JSON
export_bookmarks("json", {"include_metadata": True})
`

### Analytics
`python
# Get bookmark statistics
stats = analyze_bookmarks()
print(f"Total bookmarks: {stats['total_count']}")
`

## 🐛 Troubleshooting

### Common Issues

1. **"FastMCP not available"**
   `ash
   pip install fastmcp>=2.10.1 --upgrade
   `

2. **"Bookmark file not found"**
   - Ensure Edge is closed when running the server
   - Check Edge profile path in configuration

3. **"Import/export errors"**
   `ash
   pip install pandas jinja2
   `

### Logging

Check logs for detailed error information:
- Server logs: Console output
- File monitoring: ile_monitor.py logs
- Analytics: nalytics.py logs

## 🔧 Development

### Running Tests
`ash
python -m pytest tests/
`

### Code Structure
- **FastMCP 2.12.2** for MCP protocol implementation
- **Pydantic** for data validation
- **Watchdog** for file monitoring
- **Pandas** for data analysis
- **Jinja2** for export templates

## 📈 Performance

- **Fast startup**: ~2-3 seconds for large bookmark collections
- **Efficient search**: Indexed search for 10,000+ bookmarks
- **Low memory**: ~50MB typical usage
- **Real-time updates**: <1 second file change detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastMCP** - Excellent MCP framework
- **Microsoft Edge** - Bookmark file format compatibility
- **Anthropic** - Model Context Protocol specification

## 📞 Support

For issues and questions:
- **GitHub Issues**: https://github.com/stephanschipal/edge-bookmark-mcp/issues
- **Email**: stephanschipal@hotmail.com

---

**Built with ❤️ using FastMCP 2.12.2**



Here are all the available **edge-bookmarks MCP tools**:

🔍 **Search & Discovery:**
- `edge-bookmarks:search_bookmarks` - Advanced bookmark search with multiple modes (fuzzy, exact, semantic, domain, tags)
- `edge-bookmarks:get_search_suggestions` - Get intelligent search suggestions based on partial query

📊 **Analysis & Insights:**
- `edge-bookmarks:analyze_bookmarks` - Comprehensive bookmark collection analysis with health scoring
- `edge-bookmarks:detect_duplicates` - Advanced duplicate detection with configurable similarity threshold
- `edge-bookmarks:cleanup_duplicates` - Remove duplicate bookmarks with confirmation

📁 **Organization:**
- `edge-bookmarks:get_bookmark_folders` - Get folder structure with comprehensive statistics

📤 **Export & Backup:**
- `edge-bookmarks:export_bookmarks` - Export to various formats (JSON, CSV, HTML, XLSX, XML, YAML)

🔧 **System Management:**
- `edge-bookmarks:initialize_server` - Initialize server components with comprehensive setup
- `edge-bookmarks:get_server_status` - Get server status and health information  
- `edge-bookmarks:get_system_info` - Get comprehensive system information and capabilities
- `edge-bookmarks:get_monitoring_status` - Get detailed file monitoring status and statistics

🔄 **Real-time Operations:**
- `edge-bookmarks:subscribe_to_bookmark_changes` - Subscribe to real-time bookmark file change notifications
- `edge-bookmarks:refresh_bookmarks` - Force refresh of bookmark data and search index

**Key Features:**
- 🚀 **7,021 bookmarks** ready for search
- ⚡ **3.8ms** average search performance
- 🔍 **Multiple search modes**: fuzzy, exact, semantic, domain, tags
- 📊 **Health analysis** with duplicate detection
- 📤 **6 export formats** with advanced filtering
- 🔄 **Real-time monitoring** of bookmark changes

All tools are operational and ready to use! 🎯


Perfect! 🎉 Now that your Edge Bookmark MCP Server is working, here's how to use the tools in Claude chat:

## **Basic Tool Usage Pattern**

Simply ask Claude to perform bookmark operations naturally:

```
"List my bookmarks"
"Search for bookmarks about python"
"Find all my GitHub bookmarks" 
"Export my bookmarks to JSON"
"Show me bookmark statistics"
```

## **Specific Tool Examples**

### **📚 Browse Bookmarks**
```
"Show me all my bookmarks"
"List bookmarks in my 'Development' folder"
"What bookmarks do I have?"
```

### **🔍 Search Operations**
```
"Search for bookmarks containing 'FastMCP'"
"Find bookmarks with 'github.com' in the URL"
"Search my bookmarks for 'machine learning'"
"Look for bookmarks tagged with 'python'"
```

### **📊 Analytics & Insights**
```
"Analyze my bookmark collection"
"Show me bookmark statistics"
"Find duplicate bookmarks"
"Check for broken links in my bookmarks"
```

### **📤 Export & Backup**
```
"Export my bookmarks to CSV"
"Save my bookmarks as JSON"
"Create an HTML backup of my bookmarks"
"Export bookmarks from my 'Work' folder only"
```

### **⚙️ Server Management**
```
"Check server status"
"Refresh bookmark data"
"Show server configuration"
"Get system information"
```

## **Advanced Usage Examples**

### **Complex Searches**
```
"Search for bookmarks from last month that contain 'AI'"
"Find all bookmarks in github.com domain"
"Show me bookmarks with multiple tags"
```

### **Detailed Analysis**
```
"Give me a detailed report of my bookmark collection"
"Show me which folders have the most bookmarks"
"Find bookmarks I haven't organized yet"
```

## **Natural Language Integration**

Claude will automatically:
- ✅ **Call the right tools** based on your request
- ✅ **Format results nicely** for easy reading  
- ✅ **Combine multiple operations** if needed
- ✅ **Handle errors gracefully** and suggest alternatives

## **Example Conversation Flow**

**You:** "What bookmarks do I have about Python?"

**Claude:** *[Calls search_bookmarks("python")]* 
"I found 15 bookmarks related to Python in your collection:
- Python Documentation (docs.python.org)
- FastAPI Tutorial (fastapi.tiangolo.com)
- ..."

**You:** "Export those to a CSV file"

**Claude:** *[Calls export_bookmarks with Python search results]*
"I've exported your Python bookmarks to a CSV file..."

## **Pro Tips**

1. **Be specific** about what you want: "Search for 'react' in Development folder"
2. **Ask for combinations**: "Find GitHub bookmarks and export them"
3. **Request analysis**: "What patterns do you see in my bookmarks?"
4. **Use natural language**: No need for technical syntax!

Your Edge Bookmark MCP Server now seamlessly integrates with Claude's conversation abilities! Just ask naturally and Claude will use the appropriate tools to help manage your bookmarks.




