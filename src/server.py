#!/usr/bin/env python3
"""
Enhanced Edge Bookmark MCP Server - Production FastMCP Integration
Complete implementation with all features, error handling, and monitoring
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Try to import FastMCP, fall back gracefully
try:
    from fastmcp import FastMCP, Context
    # from fastmcp.server.elicitation import ElicitationRequest  # Not available in this FastMCP version
    from pydantic import BaseModel, Field, ValidationError
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    # Mock classes for testing without FastMCP
    class FastMCP:
        def __init__(self, **kwargs): pass
        def tool(self): return lambda f: f
        def run(self): pass
    
    class Context:
        async def info(self, msg): print(f"INFO: {msg}")
        async def error(self, msg): print(f"ERROR: {msg}")
        async def elicit(self, prompt, options): return options[0] if options else "yes"
    
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(**kwargs): return None

from config import config
from bookmark_loader import EdgeBookmarkLoader
from search_engine import BookmarkSearchEngine
from analytics import BookmarkAnalyzer
from exporter import BookmarkExporter, ExportOptions
from file_monitor import BookmarkMonitor

# Configure enhanced logging
logging.basicConfig(
    level=getattr(logging, config.log_level.value if hasattr(config, 'log_level') else 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('edge-bookmark-server.log', encoding='utf-8')
    ]
)

# Mock ElicitationRequest for compatibility
class ElicitationRequest:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

logger = logging.getLogger(__name__)

# Pydantic models for structured API responses
class BookmarkResult(BaseModel):
    """Single bookmark result with metadata."""
    title: str = Field(description="Bookmark title")
    url: str = Field(description="Bookmark URL")
    folder: str = Field(description="Folder path")
    date_added: Optional[str] = Field(description="Date added timestamp")
    score: float = Field(description="Search relevance score", ge=0.0, le=100.0)
    tags: List[str] = Field(default_factory=list, description="Bookmark tags")
    description: Optional[str] = Field(description="Bookmark description")
    profile: Optional[str] = Field(description="Edge profile name")

class SearchResults(BaseModel):
    """Search results with metadata."""
    query: str = Field(description="Original search query")
    results: List[BookmarkResult] = Field(description="Search results")
    total_found: int = Field(description="Total results found")
    search_time_ms: float = Field(description="Search execution time in milliseconds")
    search_mode: str = Field(description="Search mode used")
    suggestions: List[str] = Field(default_factory=list, description="Search suggestions")

class AnalyticsReport(BaseModel):
    """Comprehensive bookmark analytics report."""
    total_bookmarks: int = Field(description="Total bookmark count")
    duplicates_found: int = Field(description="Number of duplicates detected")
    folder_count: int = Field(description="Number of folders")
    max_folder_depth: int = Field(description="Maximum folder nesting depth")
    health_score: float = Field(description="Overall collection health score")
    organization_score: float = Field(description="Folder organization score")
    url_health_score: float = Field(description="URL quality score")
    top_domains: List[Dict[str, Any]] = Field(description="Most bookmarked domains")
    recommendations: List[str] = Field(description="Improvement recommendations")

class ServerStatus(BaseModel):
    """Server status and health information."""
    status: str = Field(description="Server status")
    bookmark_count: int = Field(description="Total bookmarks loaded")
    profile_count: int = Field(description="Number of Edge profiles")
    last_update: str = Field(description="Last update timestamp")
    uptime_seconds: float = Field(description="Server uptime in seconds")
    monitoring_active: bool = Field(description="File monitoring status")
    search_statistics: Dict[str, Any] = Field(description="Search engine statistics")
    error_count: int = Field(description="Total error count")

class ExportResult(BaseModel):
    """Export operation result."""
    success: bool = Field(description="Export success status")
    message: str = Field(description="Export result message")
    output_path: str = Field(description="Output file path")
    bookmark_count: int = Field(description="Number of bookmarks exported")
    file_size_bytes: Optional[int] = Field(description="Output file size")
    export_time_ms: float = Field(description="Export execution time")

# Initialize FastMCP server
if FASTMCP_AVAILABLE:
    mcp = FastMCP(
        name="edge-bookmark-server"
    )
else:
    mcp = FastMCP()
    logger.warning("âš ï¸ FastMCP not available, running in compatibility mode")

# Global state management
class ServerState:
    """Centralized server state management."""
    
    def __init__(self):
        self.bookmark_loader: Optional[EdgeBookmarkLoader] = None
        self.search_engine: Optional[BookmarkSearchEngine] = None
        self.analyzer: Optional[BookmarkAnalyzer] = None
        self.exporter: Optional[BookmarkExporter] = None
        self.monitor: Optional[BookmarkMonitor] = None
        
        self.start_time = datetime.now()
        self.error_count = 0
        self.is_initialized = False
        self.last_bookmark_update = None
        
        # Cache for expensive operations
        self.cached_bookmarks: List[Dict[str, Any]] = []
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl_seconds = 300  # 5 minutes
        
        logger.info("ðŸŽ¯ Server state initialized")
    
    def is_cache_valid(self) -> bool:
        """Check if bookmark cache is still valid."""
        if not self.cache_timestamp:
            return False
        return (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_ttl_seconds
    
    async def get_bookmarks(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Get bookmarks with intelligent caching."""
        if force_reload or not self.is_cache_valid():
            if self.bookmark_loader:
                self.cached_bookmarks = await self.bookmark_loader.get_cached_bookmarks()
                self.cache_timestamp = datetime.now()
                logger.info(f"ðŸ“š Refreshed bookmark cache: {len(self.cached_bookmarks)} bookmarks")
        
        return self.cached_bookmarks
    
    def increment_error(self):
        """Increment error counter."""
        self.error_count += 1

# Global server state
server_state = ServerState()

# Enhanced error handling decorator
def handle_errors(func):
    """Decorator for consistent error handling across tools."""
    async def wrapper():
        try:
            return await func()
        except ValidationError as e:
            server_state.increment_error()
            error_msg = f"Validation error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            server_state.increment_error()
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)
    return wrapper

# Server management tools
@handle_errors
@mcp.tool()
async def initialize_server(ctx: Context) -> str:
    """Initialize the bookmark server components with comprehensive setup."""
    try:
        await ctx.info("ðŸš€ Initializing Edge Bookmark MCP Server...")
        
        # Initialize all components
        server_state.bookmark_loader = EdgeBookmarkLoader()
        server_state.search_engine = BookmarkSearchEngine()
        server_state.analyzer = BookmarkAnalyzer()
        server_state.exporter = BookmarkExporter()
        
        # Load initial bookmarks
        await ctx.info("ðŸ“– Loading bookmarks from all Edge profiles...")
        bookmarks = await server_state.bookmark_loader.load_all_bookmarks()
        
        # Index bookmarks for search
        await ctx.info("ðŸ” Building search index...")
        await server_state.search_engine.index_bookmarks(bookmarks)
        
        # Initialize file monitoring
        await ctx.info("ðŸ‘ï¸ Starting file monitoring...")
        server_state.monitor = BookmarkMonitor(on_change_callback=_handle_bookmark_change)
        profiles = server_state.bookmark_loader.discovered_profiles
        await server_state.monitor.start_monitoring(profiles)
        
        # Update state
        server_state.is_initialized = True
        server_state.cached_bookmarks = bookmarks
        server_state.cache_timestamp = datetime.now()
        server_state.last_bookmark_update = datetime.now()
        
        await ctx.info(f"âœ… Server initialized successfully!")
        
        return f"""Edge Bookmark MCP Server initialized successfully!

ðŸ“Š Statistics:
- {len(bookmarks)} bookmarks loaded
- {len(profiles)} Edge profiles discovered
- Search index built with {server_state.search_engine.get_search_statistics()['total_bookmarks_indexed']} entries
- File monitoring active: {server_state.monitor.is_running}

ðŸŽ¯ Ready to serve bookmark operations!"""
        
    except Exception as e:
        server_state.increment_error()
        error_msg = f"Server initialization failed: {str(e)}"
        await ctx.error(error_msg)
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise

@handle_errors
@mcp.tool()
async def get_server_status(ctx: Context) -> ServerStatus:
    """Get comprehensive server status and health information."""
    try:
        if not server_state.is_initialized:
            return ServerStatus(
                status="not_initialized",
                bookmark_count=0,
                profile_count=0,
                last_update="Never",
                uptime_seconds=0,
                monitoring_active=False,
                search_statistics={},
                error_count=server_state.error_count
            )
        
        bookmarks = await server_state.get_bookmarks()
        profile_count = len(server_state.bookmark_loader.discovered_profiles) if server_state.bookmark_loader else 0
        uptime = (datetime.now() - server_state.start_time).total_seconds()
        
        search_stats = {}
        if server_state.search_engine:
            search_stats = server_state.search_engine.get_search_statistics()
        
        status = ServerStatus(
            status="running" if server_state.is_initialized else "initializing",
            bookmark_count=len(bookmarks),
            profile_count=profile_count,
            last_update=server_state.last_bookmark_update.isoformat() if server_state.last_bookmark_update else "Never",
            uptime_seconds=uptime,
            monitoring_active=server_state.monitor.is_running if server_state.monitor else False,
            search_statistics=search_stats,
            error_count=server_state.error_count
        )
        
        await ctx.info(f"ðŸ“Š Server status: {status.bookmark_count} bookmarks, uptime: {uptime:.0f}s")
        return status
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Status check failed: {str(e)}")
        raise

# Search and discovery tools
@handle_errors
@mcp.tool()
async def search_bookmarks(
    query: str,
    mode: str = "fuzzy",
    limit: int = 20,
    folder_filter: Optional[str] = None,
    include_suggestions: bool = True,
    ctx: Context = None
) -> SearchResults:
    """
    Advanced bookmark search with multiple modes and intelligent suggestions.
    
    Args:
        query: Search query text
        mode: Search mode ('fuzzy', 'exact', 'semantic', 'domain', 'tags')
        limit: Maximum results to return (1-100)
        folder_filter: Optional folder path filter
        include_suggestions: Include search suggestions in response
    """
    if not server_state.search_engine:
        raise ValueError("Search engine not initialized. Run initialize_server first.")
    
    # Validate inputs
    limit = max(1, min(limit, config.search.max_limit))
    valid_modes = ['fuzzy', 'exact', 'semantic', 'domain', 'tags']
    if mode not in valid_modes:
        raise ValueError(f"Invalid search mode. Use one of: {valid_modes}")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        await ctx.info(f"ðŸ” Searching for '{query}' (mode: {mode}, limit: {limit})")
        
        # Perform search based on mode
        if mode == "fuzzy":
            results = await server_state.search_engine.fuzzy_search(query, limit, folder_filter)
        elif mode == "exact":
            results = await server_state.search_engine.exact_search(query, limit, folder_filter)
        elif mode == "semantic":
            results = await server_state.search_engine.semantic_search(query, limit, folder_filter)
        elif mode == "domain":
            results = await server_state.search_engine.search_by_domain(query, limit)
        elif mode == "tags":
            tag_list = [tag.strip() for tag in query.split(',')]
            results = await server_state.search_engine.search_by_tags(tag_list, match_all=False, limit=limit)
        else:
            raise ValueError(f"Search mode '{mode}' not implemented")
        
        search_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Convert to structured format
        bookmark_results = [
            BookmarkResult(
                title=r['bookmark']['title'],
                url=r['bookmark']['url'],
                folder=r['bookmark'].get('folder_path', ''),
                date_added=r['bookmark'].get('date_added'),
                score=r['score'],
                tags=r['bookmark'].get('tags', []),
                description=r['bookmark'].get('description'),
                profile=r['bookmark'].get('profile')
            )
            for r in results
        ]
        
        # Get search suggestions if requested
        suggestions = []
        if include_suggestions and len(query) >= 2:
            suggestions = await server_state.search_engine.get_suggestions(query, limit=5)
        
        search_results = SearchResults(
            query=query,
            results=bookmark_results,
            total_found=len(bookmark_results),
            search_time_ms=round(search_time, 2),
            search_mode=mode,
            suggestions=suggestions
        )
        
        await ctx.info(f"âœ… Found {len(bookmark_results)} results in {search_time:.1f}ms")
        return search_results
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Search failed: {str(e)}")
        raise

@handle_errors
@mcp.tool()
async def get_search_suggestions(partial_query: str, limit: int = 10, ctx: Context = None) -> List[str]:
    """Get intelligent search suggestions based on partial query and search history."""
    if not server_state.search_engine:
        raise ValueError("Search engine not initialized")
    
    try:
        suggestions = await server_state.search_engine.get_suggestions(partial_query, limit)
        await ctx.info(f"ðŸ’¡ Generated {len(suggestions)} suggestions for '{partial_query}'")
        return suggestions
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Suggestion generation failed: {str(e)}")
        raise

# Analytics and insights tools
@handle_errors
@mcp.tool()
async def analyze_bookmarks(
    include_duplicates: bool = True,
    include_health_score: bool = True,
    include_folder_analysis: bool = True,
    ctx: Context = None
) -> AnalyticsReport:
    """
    Comprehensive bookmark collection analysis with customizable scope.
    
    Args:
        include_duplicates: Analyze duplicate bookmarks
        include_health_score: Calculate collection health score
        include_folder_analysis: Analyze folder organization
    """
    if not server_state.analyzer:
        raise ValueError("Analytics engine not initialized")
    
    try:
        await ctx.info("ðŸ“Š Starting comprehensive bookmark analysis...")
        
        bookmarks = await server_state.get_bookmarks()
        analysis = await server_state.analyzer.analyze_bookmarks(bookmarks)
        
        # Extract key metrics
        duplicates_count = len(analysis['duplicates']['duplicate_matches']) if include_duplicates else 0
        folder_analysis = analysis.get('folder_structure', {})
        health_score = analysis.get('health_score', {})
        url_health = analysis.get('url_health', {})
        domain_analysis = analysis.get('domain_analysis', {})
        
        # Build structured report
        report = AnalyticsReport(
            total_bookmarks=analysis['metadata']['total_bookmarks'],
            duplicates_found=duplicates_count,
            folder_count=folder_analysis.get('total_folders', 0),
            max_folder_depth=folder_analysis.get('max_depth', 0),
            health_score=health_score.overall_score if hasattr(health_score, 'overall_score') else 0.0,
            organization_score=health_score.organization_score if hasattr(health_score, 'organization_score') else 0.0,
            url_health_score=url_health.get('health_score', 100.0),
            top_domains=domain_analysis.get('top_domains', [])[:10],
            recommendations=analysis.get('recommendations', [])
        )
        
        await ctx.info(f"âœ… Analysis complete: {report.duplicates_found} duplicates, {report.health_score:.1f}% health score")
        return report
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Analysis failed: {str(e)}")
        raise

@handle_errors
@mcp.tool()
async def detect_duplicates(
    similarity_threshold: float = 80.0,
    show_details: bool = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Advanced duplicate detection with configurable similarity threshold.
    
    Args:
        similarity_threshold: Minimum similarity score for duplicates (0-100)
        show_details: Include detailed duplicate information
    """
    if not server_state.analyzer:
        raise ValueError("Analytics engine not initialized")
    
    try:
        await ctx.info(f"ðŸ” Detecting duplicates with {similarity_threshold}% threshold...")
        
        bookmarks = await server_state.get_bookmarks()
        
        # Temporarily adjust threshold in config
        original_threshold = config.analytics.duplicate_similarity_threshold
        config.analytics.duplicate_similarity_threshold = similarity_threshold
        
        duplicates = await server_state.analyzer.detect_duplicates(bookmarks)
        
        # Restore original threshold
        config.analytics.duplicate_similarity_threshold = original_threshold
        
        result = {
            'total_duplicates': len(duplicates),
            'threshold_used': similarity_threshold,
            'summary': f"Found {len(duplicates)} potential duplicates"
        }
        
        if show_details:
            result['duplicates'] = duplicates[:50]  # Limit for performance
        
        await ctx.info(f"ðŸŽ¯ Found {len(duplicates)} potential duplicates")
        return result
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Duplicate detection failed: {str(e)}")
        raise

@handle_errors
@mcp.tool()
async def cleanup_duplicates(
    auto_confirm: bool = False,
    similarity_threshold: float = 90.0,
    ctx: Context = None
) -> str:
    """
    Remove duplicate bookmarks with optional confirmation and high similarity threshold.
    
    Args:
        auto_confirm: Skip confirmation dialog
        similarity_threshold: Minimum similarity for auto-removal (recommended: 90+)
    """
    if not server_state.analyzer:
        raise ValueError("Analytics engine not initialized")
    
    try:
        bookmarks = await server_state.get_bookmarks()
        
        # Use high threshold for safety
        config.analytics.duplicate_similarity_threshold = max(similarity_threshold, 85.0)
        duplicates = await server_state.analyzer.detect_duplicates(bookmarks)
        
        if not duplicates:
            await ctx.info("âœ… No duplicates found with current threshold")
            return "No duplicate bookmarks found with the specified similarity threshold."
        
        high_confidence = [d for d in duplicates if d.get('similarity_score', 0) >= 95.0]
        
        if not auto_confirm:
            action = await ctx.elicit(
                prompt=f"Found {len(duplicates)} duplicates ({len(high_confidence)} high-confidence). Proceed with cleanup?",
                options=["yes", "no", "show_details", "high_confidence_only"]
            )
            
            if action == "no":
                return "Duplicate cleanup cancelled by user."
            elif action == "show_details":
                details = "\n".join([
                    f"- {dup.get('original', {}).get('title', 'Unknown')} vs {dup.get('duplicate', {}).get('title', 'Unknown')} ({dup.get('similarity_score', 0):.1f}%)"
                    for dup in duplicates[:10]
                ])
                await ctx.info(f"Sample duplicates:\n{details}")
                
                confirm = await ctx.elicit(
                    prompt="Proceed with cleanup after reviewing details?",
                    options=["yes", "no"]
                )
                if confirm == "no":
                    return "Duplicate cleanup cancelled after review."
            elif action == "high_confidence_only":
                duplicates = high_confidence
        
        # Simulate cleanup (in production, this would modify actual bookmark files)
        removed_count = len(duplicates)
        await server_state.analyzer.remove_duplicates(duplicates)
        
        # Refresh search index
        updated_bookmarks = await server_state.get_bookmarks(force_reload=True)
        if server_state.search_engine:
            await server_state.search_engine.index_bookmarks(updated_bookmarks)
        
        await ctx.info(f"âœ… Cleanup simulation completed")
        return f"Successfully identified {removed_count} duplicates for removal (simulation mode)."
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Cleanup failed: {str(e)}")
        raise

# Export and backup tools
@handle_errors
@mcp.tool()
async def export_bookmarks(
    format: str,
    output_file: str,
    filter_folder: Optional[str] = None,
    include_metadata: bool = True,
    sort_by: str = "title",
    ctx: Context = None
) -> ExportResult:
    """
    Export bookmarks to various formats with advanced filtering and sorting.
    
    Args:
        format: Export format ('json', 'csv', 'html', 'xlsx', 'xml', 'yaml')
        output_file: Output file path
        filter_folder: Optional folder to export (exports all if None)
        include_metadata: Include extended metadata in export
        sort_by: Sort bookmarks by field ('title', 'url', 'date_added', 'folder')
    """
    if not server_state.exporter:
        raise ValueError("Export engine not initialized")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        await ctx.info(f"ðŸ“¤ Starting export to {format} format...")
        
        bookmarks = await server_state.get_bookmarks()
        
        # Apply folder filter if specified
        if filter_folder:
            original_count = len(bookmarks)
            bookmarks = [b for b in bookmarks if b.get('folder_path', '').startswith(filter_folder)]
            await ctx.info(f"ðŸ“ Filtered to {len(bookmarks)} bookmarks from folder '{filter_folder}' (was {original_count})")
        
        # Setup export options
        options = ExportOptions(
            include_metadata=include_metadata,
            sort_by=sort_by,
            sort_ascending=True
        )
        
        # Perform export
        result_message = await server_state.exporter.export_bookmarks(
            bookmarks, format, output_file, options
        )
        
        export_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Get file size if possible
        file_size = None
        try:
            output_path = Path(output_file)
            if output_path.exists():
                file_size = output_path.stat().st_size
        except:
            pass
        
        export_result = ExportResult(
            success=True,
            message=result_message,
            output_path=str(Path(output_file).absolute()),
            bookmark_count=len(bookmarks),
            file_size_bytes=file_size,
            export_time_ms=round(export_time, 2)
        )
        
        await ctx.info(f"âœ… Export completed: {result_message}")
        return export_result
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Export failed: {str(e)}")
        
        return ExportResult(
            success=False,
            message=f"Export failed: {str(e)}",
            output_path=output_file,
            bookmark_count=0,
            file_size_bytes=None,
            export_time_ms=0
        )

# Folder and organization tools
@handle_errors
@mcp.tool()
async def get_bookmark_folders(
    include_stats: bool = True,
    sort_by: str = "bookmark_count",
    ctx: Context = None
) -> List[Dict[str, Any]]:
    """
    Get bookmark folder structure with comprehensive statistics.
    
    Args:
        include_stats: Include detailed folder statistics
        sort_by: Sort folders by ('name', 'bookmark_count', 'depth')
    """
    try:
        bookmarks = await server_state.get_bookmarks()
        folder_stats = {}
        
        for bookmark in bookmarks:
            folder = bookmark.get('folder_path', 'Root')
            if folder not in folder_stats:
                folder_stats[folder] = {
                    'count': 0,
                    'bookmarks': [],
                    'domains': set(),
                    'tags': set()
                }
            
            stats = folder_stats[folder]
            stats['count'] += 1
            stats['bookmarks'].append(bookmark.get('title', 'Untitled'))
            
            # Collect domains and tags if including stats
            if include_stats:
                url = bookmark.get('url', '')
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            stats['domains'].add(domain)
                    except:
                        pass
                
                for tag in bookmark.get('tags', []):
                    stats['tags'].add(tag)
        
        # Build folder list
        folders = []
        for folder, stats in folder_stats.items():
            folder_info = {
                'folder': folder,
                'bookmark_count': stats['count'],
                'depth': len(folder.split('/')) if folder != 'Root' else 0,
                'sample_bookmarks': stats['bookmarks'][:5]
            }
            
            if include_stats:
                folder_info.update({
                    'unique_domains': len(stats['domains']),
                    'unique_tags': len(stats['tags']),
                    'avg_bookmarks_per_subfolder': stats['count']  # Simplified calculation
                })
            
            folders.append(folder_info)
        
        # Sort folders
        sort_keys = {
            'name': lambda x: x['folder'].lower(),
            'bookmark_count': lambda x: x['bookmark_count'],
            'depth': lambda x: x['depth']
        }
        
        if sort_by in sort_keys:
            folders.sort(key=sort_keys[sort_by], reverse=(sort_by == 'bookmark_count'))
        
        await ctx.info(f"ðŸ“ Found {len(folders)} folders with {len(bookmarks)} total bookmarks")
        return folders
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Folder analysis failed: {str(e)}")
        raise

# File monitoring and real-time updates
@handle_errors
@mcp.tool()
async def get_monitoring_status(ctx: Context = None) -> Dict[str, Any]:
    """Get detailed file monitoring status and statistics."""
    try:
        if not server_state.monitor:
            return {
                'monitoring_active': False,
                'message': 'File monitoring not initialized'
            }
        
        status = server_state.monitor.get_monitoring_status()
        await ctx.info(f"ðŸ‘ï¸ Monitoring status: {status['monitoring_method']}, {status['subscriber_count']} subscribers")
        return status
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Monitoring status check failed: {str(e)}")
        raise

@handle_errors
@mcp.tool()
async def subscribe_to_bookmark_changes(ctx: Context) -> str:
    """Subscribe to real-time bookmark file change notifications."""
    try:
        if not server_state.monitor:
            raise ValueError("File monitoring not initialized")
        
        session_id = f"session_{datetime.now().timestamp()}"
        server_state.monitor.add_subscriber(session_id)
        
        await ctx.info(f"ðŸ“¡ Subscribed to bookmark changes: {session_id}")
        return f"Successfully subscribed to real-time bookmark updates. Session ID: {session_id}"
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Subscription failed: {str(e)}")
        raise

# System maintenance and utilities
@handle_errors
@mcp.tool()
async def refresh_bookmarks(ctx: Context = None) -> str:
    """Force refresh of bookmark data and search index."""
    try:
        await ctx.info("ðŸ”„ Refreshing bookmark data...")
        
        if server_state.bookmark_loader:
            bookmarks = await server_state.bookmark_loader.reload_bookmarks()
            server_state.cached_bookmarks = bookmarks
            server_state.cache_timestamp = datetime.now()
            server_state.last_bookmark_update = datetime.now()
            
            # Rebuild search index
            if server_state.search_engine:
                await server_state.search_engine.index_bookmarks(bookmarks)
            
            await ctx.info(f"âœ… Refreshed {len(bookmarks)} bookmarks and search index")
            return f"Successfully refreshed {len(bookmarks)} bookmarks and rebuilt search index."
        else:
            raise ValueError("Bookmark loader not initialized")
            
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"Refresh failed: {str(e)}")
        raise

@handle_errors
@mcp.tool()
async def get_system_info(ctx: Context = None) -> Dict[str, Any]:
    """Get comprehensive system information and capabilities."""
    try:
        import sys
        import platform
        
        # Component availability
        capabilities = {
            'fastmcp_available': FASTMCP_AVAILABLE,
            'rapidfuzz_available': getattr(server_state.search_engine, 'RAPIDFUZZ_AVAILABLE', False) if server_state.search_engine else False,
            'watchdog_available': getattr(server_state.monitor, 'WATCHDOG_AVAILABLE', False) if server_state.monitor else False,
            'pandas_available': getattr(server_state.exporter, 'PANDAS_AVAILABLE', False) if server_state.exporter else False,
            'real_time_monitoring': server_state.monitor.is_running if server_state.monitor else False
        }
        
        # System info
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'server_version': '1.0.0',
            'server_uptime_seconds': (datetime.now() - server_state.start_time).total_seconds(),
            'initialization_status': server_state.is_initialized,
            'total_errors': server_state.error_count,
            'capabilities': capabilities
        }
        
        # Component statistics
        if server_state.search_engine:
            system_info['search_statistics'] = server_state.search_engine.get_search_statistics()
        
        if server_state.exporter:
            system_info['export_statistics'] = server_state.exporter.get_export_statistics()
        
        await ctx.info("ðŸ“‹ System information compiled")
        return system_info
        
    except Exception as e:
        server_state.increment_error()
        await ctx.error(f"System info failed: {str(e)}")
        raise

# Event handlers
async def _handle_bookmark_change(file_path: str, event_type: str = "modified"):
    """Handle bookmark file changes for real-time updates."""
    try:
        logger.info(f"ðŸ“ Bookmark file {event_type}: {file_path}")
        
        # Update cache timestamp to trigger refresh on next access
        server_state.cache_timestamp = None
        server_state.last_bookmark_update = datetime.now()
        
        # Reload and reindex if components are available
        if server_state.bookmark_loader and server_state.search_engine:
            bookmarks = await server_state.bookmark_loader.reload_bookmarks()
            await server_state.search_engine.index_bookmarks(bookmarks)
            server_state.cached_bookmarks = bookmarks
            server_state.cache_timestamp = datetime.now()
            
            logger.info(f"ðŸ”„ Reindexed {len(bookmarks)} bookmarks after file change")
            
    except Exception as e:
        server_state.increment_error()
        logger.error(f"Error handling bookmark change: {e}")

def main():
    """Main entry point for the server."""
    try:
        logger.info("ðŸš€ Starting Edge Bookmark MCP Server...")
        logger.info(f"FastMCP available: {FASTMCP_AVAILABLE}")
        logger.info(f"Configuration loaded: {config.server_name} v{config.server_version}")
        
        if FASTMCP_AVAILABLE:
            mcp.run()
        else:
            logger.warning("âš ï¸ FastMCP not available - server running in compatibility mode")
            logger.info("Install FastMCP with: pip install fastmcp")
            
            # Keep server alive for testing
            import time
            while True:
                time.sleep(60)
                logger.info("ðŸ“Š Server heartbeat (compatibility mode)")
                
    except KeyboardInterrupt:
        logger.info("ðŸ‘‹ Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup
        if server_state.monitor:
            asyncio.run(server_state.monitor.stop_monitoring())
        logger.info("ðŸ Server stopped")

if __name__ == "__main__":
    main()
