"""
Edge Bookmark Loader - Safe loading and parsing of Edge bookmark files
Handles multiple profiles and provides safe file operations.
"""

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import aiofiles
import psutil

logger = logging.getLogger(__name__)

class EdgeBookmarkLoader:
    """Loads and manages Microsoft Edge bookmark files across all profiles."""
    
    def __init__(self):
        self.discovered_profiles: List[Path] = []
        self.cached_bookmarks: List[Dict[str, Any]] = []
        self.last_load_time: Optional[datetime] = None
        self.backup_directory = Path("data/backups")
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
    async def discover_edge_profiles(self) -> List[Path]:
        """Discover all Edge profile bookmark files."""
        bookmark_files = []
        
        # Edge user data base path
        edge_base = Path.home() / "AppData/Local/Microsoft/Edge/User Data"
        
        if not edge_base.exists():
            logger.warning(f"Edge user data directory not found: {edge_base}")
            return bookmark_files
        
        # Check default profile
        default_bookmarks = edge_base / "Default/Bookmarks"
        if default_bookmarks.exists():
            bookmark_files.append(default_bookmarks)
            logger.info(f"Found default profile bookmarks: {default_bookmarks}")
        
        # Check additional profiles (Profile 1, Profile 2, etc.)
        for profile_dir in edge_base.glob("Profile *"):
            if profile_dir.is_dir():
                profile_bookmarks = profile_dir / "Bookmarks"
                if profile_bookmarks.exists():
                    bookmark_files.append(profile_bookmarks)
                    logger.info(f"Found profile bookmarks: {profile_bookmarks}")
        
        # Check Guest profile
        guest_bookmarks = edge_base / "Guest Profile/Bookmarks"
        if guest_bookmarks.exists():
            bookmark_files.append(guest_bookmarks)
            logger.info(f"Found guest profile bookmarks: {guest_bookmarks}")
        
        self.discovered_profiles = bookmark_files
        logger.info(f"Discovered {len(bookmark_files)} Edge profile(s)")
        return bookmark_files
    
    async def ensure_edge_closed(self) -> bool:
        """Ensure Edge is closed before file operations to prevent corruption."""
        edge_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'msedge' in proc.info['name'].lower():
                    edge_processes.append(proc)
            
            if not edge_processes:
                return True
            
            logger.info(f"Found {len(edge_processes)} Edge processes, attempting to close...")
            
            # Try graceful termination first
            for proc in edge_processes:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Wait for processes to terminate
            await asyncio.sleep(2)
            
            # Force kill if still running
            for proc in edge_processes:
                try:
                    if proc.is_running():
                        proc.kill()
                        logger.warning(f"Force killed Edge process {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Final wait
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"Error closing Edge processes: {e}")
            return False
    
    async def create_backup(self, bookmark_file: Path) -> Path:
        """Create timestamped backup of bookmark file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_name = bookmark_file.parent.name
        backup_name = f"Bookmarks_{profile_name}_{timestamp}.json"
        backup_path = self.backup_directory / backup_name
        
        try:
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy with metadata preservation
            shutil.copy2(bookmark_file, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    async def load_bookmark_file(self, file_path: Path) -> Dict[str, Any]:
        """Safely load a single bookmark file with error handling."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Ensure Edge is closed
                await self.ensure_edge_closed()
                
                # Wait a bit for file system to stabilize
                if attempt > 0:
                    await asyncio.sleep(retry_delay * attempt)
                
                # Read file asynchronously
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                # Parse JSON
                data = json.loads(content)
                logger.info(f"Successfully loaded bookmarks from {file_path}")
                return data
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {e}")
                
                # Try backup file if available
                backup_files = list(self.backup_directory.glob(f"Bookmarks_{file_path.parent.name}_*.json"))
                if backup_files:
                    latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Attempting to load from backup: {latest_backup}")
                    try:
                        async with aiofiles.open(latest_backup, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        return json.loads(content)
                    except Exception as backup_error:
                        logger.error(f"Backup load failed: {backup_error}")
                
                if attempt == max_retries - 1:
                    raise
                    
            except FileNotFoundError:
                logger.error(f"Bookmark file not found: {file_path}")
                raise
                
            except PermissionError as e:
                logger.error(f"Permission denied accessing {file_path}: {e}")
                if attempt == max_retries - 1:
                    raise
                
            except Exception as e:
                logger.error(f"Unexpected error loading {file_path}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        raise Exception(f"Failed to load bookmark file after {max_retries} attempts")
    
    def extract_bookmarks_recursive(self, node: Dict[str, Any], folder_path: str = "") -> List[Dict[str, Any]]:
        """Recursively extract bookmarks from Edge's nested JSON structure."""
        bookmarks = []
        
        if node.get('type') == 'url':
            # This is a bookmark
            bookmark = {
                'id': node.get('id', ''),
                'title': node.get('name', ''),
                'url': node.get('url', ''),
                'folder_path': folder_path,
                'date_added': self._convert_chrome_timestamp(node.get('date_added')),
                'date_modified': self._convert_chrome_timestamp(node.get('date_modified')),
                'meta_info': node.get('meta_info', {}),
                'guid': node.get('guid', ''),
                'tags': self._extract_tags_from_meta(node.get('meta_info', {}))
            }
            bookmarks.append(bookmark)
            
        elif node.get('type') == 'folder' and 'children' in node:
            # This is a folder with children
            folder_name = node.get('name', 'Unnamed Folder')
            new_folder_path = f"{folder_path}/{folder_name}" if folder_path else folder_name
            
            for child in node['children']:
                bookmarks.extend(self.extract_bookmarks_recursive(child, new_folder_path))
        
        return bookmarks
    
    def _convert_chrome_timestamp(self, timestamp: Optional[str]) -> Optional[str]:
        """Convert Chrome/Edge timestamp to ISO format."""
        if not timestamp:
            return None
        
        try:
            # Chrome timestamps are microseconds since Windows epoch (1601-01-01)
            # Convert to Unix timestamp then to datetime
            timestamp_int = int(timestamp)
            windows_epoch = datetime(1601, 1, 1)
            unix_epoch = datetime(1970, 1, 1)
            epoch_delta = (unix_epoch - windows_epoch).total_seconds()
            
            unix_timestamp = (timestamp_int / 1_000_000) - epoch_delta
            dt = datetime.fromtimestamp(unix_timestamp)
            return dt.isoformat()
            
        except (ValueError, OverflowError) as e:
            logger.warning(f"Invalid timestamp conversion: {timestamp} - {e}")
            return None
    
    def _extract_tags_from_meta(self, meta_info: Dict[str, Any]) -> List[str]:
        """Extract tags from bookmark meta information."""
        tags = []
        
        # Check various possible tag locations in Edge metadata
        if 'tags' in meta_info:
            tags.extend(meta_info['tags'])
        
        if 'keywords' in meta_info:
            tags.extend(meta_info['keywords'])
        
        # Clean and deduplicate tags
        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                cleaned_tags.append(tag.strip().lower())
        
        return list(set(cleaned_tags))
    
    async def load_all_bookmarks(self) -> List[Dict[str, Any]]:
        """Load bookmarks from all discovered Edge profiles."""
        if not self.discovered_profiles:
            await self.discover_edge_profiles()
        
        if not self.discovered_profiles:
            logger.warning("No Edge bookmark files found")
            return []
        
        all_bookmarks = []
        
        for profile_path in self.discovered_profiles:
            try:
                logger.info(f"Loading bookmarks from profile: {profile_path.parent.name}")
                
                # Create backup before reading
                await self.create_backup(profile_path)
                
                # Load bookmark data
                bookmark_data = await self.load_bookmark_file(profile_path)
                
                # Extract bookmarks from roots
                if 'roots' in bookmark_data:
                    profile_bookmarks = []
                    
                    for root_name, root_data in bookmark_data['roots'].items():
                        if isinstance(root_data, dict) and 'children' in root_data:
                            folder_bookmarks = self.extract_bookmarks_recursive(
                                root_data, root_name
                            )
                            profile_bookmarks.extend(folder_bookmarks)
                    
                    # Add profile information to each bookmark
                    for bookmark in profile_bookmarks:
                        bookmark['profile'] = profile_path.parent.name
                        bookmark['profile_path'] = str(profile_path.parent)
                    
                    all_bookmarks.extend(profile_bookmarks)
                    logger.info(f"Loaded {len(profile_bookmarks)} bookmarks from {profile_path.parent.name}")
                
            except Exception as e:
                logger.error(f"Failed to load bookmarks from {profile_path}: {e}")
                continue
        
        # Cache results
        self.cached_bookmarks = all_bookmarks
        self.last_load_time = datetime.now()
        
        logger.info(f"Total bookmarks loaded: {len(all_bookmarks)} from {len(self.discovered_profiles)} profiles")
        return all_bookmarks
    
    async def get_cached_bookmarks(self) -> List[Dict[str, Any]]:
        """Get cached bookmarks, loading if necessary."""
        if not self.cached_bookmarks or not self.last_load_time:
            return await self.load_all_bookmarks()
        
        return self.cached_bookmarks
    
    async def reload_bookmarks(self) -> List[Dict[str, Any]]:
        """Force reload of all bookmarks."""
        logger.info("Forcing bookmark reload...")
        self.cached_bookmarks = []
        self.last_load_time = None
        return await self.load_all_bookmarks()
    
    async def validate_bookmark_files(self) -> Dict[str, Any]:
        """Validate all bookmark files and return health status."""
        validation_results = {
            'total_profiles': len(self.discovered_profiles),
            'valid_files': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'file_details': []
        }
        
        for profile_path in self.discovered_profiles:
            file_info = {
                'profile': profile_path.parent.name,
                'path': str(profile_path),
                'status': 'unknown',
                'size_bytes': 0,
                'bookmark_count': 0,
                'last_modified': None,
                'error': None
            }
            
            try:
                if not profile_path.exists():
                    file_info['status'] = 'missing'
                    validation_results['missing_files'] += 1
                else:
                    stat = profile_path.stat()
                    file_info['size_bytes'] = stat.st_size
                    file_info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                    
                    # Try to load and validate
                    bookmark_data = await self.load_bookmark_file(profile_path)
                    if 'roots' in bookmark_data:
                        bookmarks = []
                        for root_data in bookmark_data['roots'].values():
                            if isinstance(root_data, dict):
                                bookmarks.extend(self.extract_bookmarks_recursive(root_data))
                        
                        file_info['bookmark_count'] = len(bookmarks)
                        file_info['status'] = 'valid'
                        validation_results['valid_files'] += 1
                    else:
                        file_info['status'] = 'invalid_format'
                        validation_results['corrupted_files'] += 1
                        
            except Exception as e:
                file_info['status'] = 'error'
                file_info['error'] = str(e)
                validation_results['corrupted_files'] += 1
                logger.error(f"Validation error for {profile_path}: {e}")
            
            validation_results['file_details'].append(file_info)
        
        return validation_results
    
    async def get_profile_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about each profile."""
        if not self.cached_bookmarks:
            await self.load_all_bookmarks()
        
        profile_stats = {}
        
        for bookmark in self.cached_bookmarks:
            profile = bookmark.get('profile', 'Unknown')
            
            if profile not in profile_stats:
                profile_stats[profile] = {
                    'bookmark_count': 0,
                    'folder_count': 0,
                    'folders': set(),
                    'domains': set(),
                    'date_range': {'earliest': None, 'latest': None}
                }
            
            stats = profile_stats[profile]
            stats['bookmark_count'] += 1
            
            # Track folders
            folder = bookmark.get('folder_path', '')
            if folder:
                stats['folders'].add(folder)
            
            # Track domains
            url = bookmark.get('url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain:
                        stats['domains'].add(domain)
                except:
                    pass
            
            # Track date range
            date_added = bookmark.get('date_added')
            if date_added:
                if not stats['date_range']['earliest'] or date_added < stats['date_range']['earliest']:
                    stats['date_range']['earliest'] = date_added
                if not stats['date_range']['latest'] or date_added > stats['date_range']['latest']:
                    stats['date_range']['latest'] = date_added
        
        # Convert sets to counts
        for profile, stats in profile_stats.items():
            stats['folder_count'] = len(stats['folders'])
            stats['unique_domains'] = len(stats['domains'])
            del stats['folders']  # Remove sets for JSON serialization
            del stats['domains']
        
        return profile_stats