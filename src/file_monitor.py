"""
Advanced File Monitor for Edge Bookmark MCP Server
Real-time monitoring of Edge bookmark files with intelligent debouncing
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime
from collections import defaultdict

# Try to import watchdog, fall back to polling
try:
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    PatternMatchingEventHandler = None
    FileSystemEvent = None

from config import config

logger = logging.getLogger(__name__)

class BookmarkFileHandler(PatternMatchingEventHandler):
    """Event handler for bookmark file changes."""
    
    def __init__(self, callback: Callable, debounce_seconds: float = 2.0):
        # Patterns to watch for bookmark files
        patterns = ["**/Bookmarks", "**/Bookmarks.bak", "**/bookmarks.json"]
        ignore_patterns = [
            "**/tmp/*", "**/.tmp*", "**/lock*", "**/~*",
            "**/Bookmarks.tmp", "**/Bookmarks.lock"
        ]
        
        super().__init__(
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            ignore_directories=True,
            case_sensitive=False
        )
        
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.last_event_time: Dict[str, float] = {}
        self.pending_events: Dict[str, asyncio.Task] = {}
        
        logger.info(f"📁 BookmarkFileHandler initialized with {debounce_seconds}s debounce")
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        self._handle_event(event, "modified")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        self._handle_event(event, "created")
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move events."""
        if event.is_directory:
            return
        
        # Handle both source and destination
        if hasattr(event, 'src_path'):
            self._handle_event(event, "moved_from", event.src_path)
        if hasattr(event, 'dest_path'):
            self._handle_event(event, "moved_to", event.dest_path)
    
    def _handle_event(self, event: FileSystemEvent, event_type: str, file_path: str = None):
        """Handle file system events with debouncing."""
        file_path = file_path or event.src_path
        now = time.time()
        
        # Check if we should debounce this event
        if file_path in self.last_event_time:
            time_since_last = now - self.last_event_time[file_path]
            if time_since_last < self.debounce_seconds:
                logger.debug(f"🔄 Debouncing event for {file_path} ({time_since_last:.1f}s)")
                
                # Cancel pending event if exists
                if file_path in self.pending_events:
                    self.pending_events[file_path].cancel()
                
                # Schedule new debounced event
                task = asyncio.create_task(self._debounced_callback(file_path, event_type))
                self.pending_events[file_path] = task
                return
        
        self.last_event_time[file_path] = now
        
        # Process immediately if no recent events
        logger.info(f"📝 File {event_type}: {file_path}")
        asyncio.create_task(self.callback(file_path, event_type))
    
    async def _debounced_callback(self, file_path: str, event_type: str):
        """Execute callback after debounce delay."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            logger.info(f"📝 Debounced file {event_type}: {file_path}")
            await self.callback(file_path, event_type)
        except asyncio.CancelledError:
            logger.debug(f"🚫 Debounced event cancelled for {file_path}")
        finally:
            # Clean up
            if file_path in self.pending_events:
                del self.pending_events[file_path]

class PollingMonitor:
    """Fallback polling monitor when watchdog is not available."""
    
    def __init__(self, paths: List[Path], callback: Callable, poll_interval: float = 5.0):
        self.paths = paths
        self.callback = callback
        self.poll_interval = poll_interval
        self.file_states: Dict[str, Dict] = {}
        self.running = False
        self.poll_task: Optional[asyncio.Task] = None
        
        logger.info(f"🔄 PollingMonitor initialized for {len(paths)} paths")
        self._initialize_file_states()
    
    def _initialize_file_states(self):
        """Initialize file state tracking."""
        for path in self.paths:
            if path.exists():
                stat = path.stat()
                self.file_states[str(path)] = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'exists': True
                }
            else:
                self.file_states[str(path)] = {
                    'mtime': 0,
                    'size': 0,
                    'exists': False
                }
    
    async def start(self):
        """Start polling."""
        self.running = True
        self.poll_task = asyncio.create_task(self._poll_loop())
        logger.info("🔄 Polling monitor started")
    
    async def stop(self):
        """Stop polling."""
        self.running = False
        if self.poll_task:
            self.poll_task.cancel()
            try:
                await self.poll_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Polling monitor stopped")
    
    async def _poll_loop(self):
        """Main polling loop."""
        try:
            while self.running:
                await self._check_files()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.debug("🔄 Polling loop cancelled")
    
    async def _check_files(self):
        """Check files for changes."""
        for path_str in list(self.file_states.keys()):
            path = Path(path_str)
            old_state = self.file_states[path_str]
            
            if path.exists():
                stat = path.stat()
                new_state = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'exists': True
                }
                
                # Check for changes
                if not old_state['exists']:
                    # File was created
                    logger.info(f"📝 File created (polling): {path}")
                    await self.callback(str(path), "created")
                elif (old_state['mtime'] != new_state['mtime'] or 
                      old_state['size'] != new_state['size']):
                    # File was modified
                    logger.info(f"📝 File modified (polling): {path}")
                    await self.callback(str(path), "modified")
                
                self.file_states[path_str] = new_state
            else:
                if old_state['exists']:
                    # File was deleted
                    logger.info(f"📝 File deleted (polling): {path}")
                    await self.callback(str(path), "deleted")
                    self.file_states[path_str] = {
                        'mtime': 0,
                        'size': 0,
                        'exists': False
                    }

class BookmarkMonitor:
    """Advanced bookmark file monitor with multiple strategies."""
    
    def __init__(self, on_change_callback: Optional[Callable] = None):
        self.on_change_callback = on_change_callback
        self.is_running = False
        self.observer: Optional[Observer] = None
        self.polling_monitor: Optional[PollingMonitor] = None
        self.monitored_paths: Set[Path] = set()
        self.subscribers: Dict[str, Dict] = {}
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'last_event_time': None,
            'monitoring_method': 'none',
            'start_time': None,
            'error_count': 0
        }
        
        logger.info(f"👁️ BookmarkMonitor initialized (watchdog: {WATCHDOG_AVAILABLE})")
    
    async def start_monitoring(self, edge_profiles: Optional[List[Path]] = None) -> None:
        """Start monitoring bookmark files."""
        logger.info("🚀 Starting bookmark file monitoring...")
        
        if edge_profiles:
            self.monitored_paths.update(edge_profiles)
        else:
            # Auto-discover Edge profiles
            await self._discover_bookmark_files()
        
        if not self.monitored_paths:
            logger.warning("⚠️ No bookmark files found to monitor")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Try watchdog first, fall back to polling
        if WATCHDOG_AVAILABLE:
            await self._start_watchdog_monitoring()
        else:
            await self._start_polling_monitoring()
        
        logger.info(f"✅ Monitoring started for {len(self.monitored_paths)} bookmark files")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        logger.info("⏹️ Stopping bookmark file monitoring...")
        
        self.is_running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.observer = None
        
        if self.polling_monitor:
            await self.polling_monitor.stop()
            self.polling_monitor = None
        
        logger.info("✅ Monitoring stopped")
    
    async def _discover_bookmark_files(self) -> None:
        """Auto-discover Edge bookmark files."""
        logger.info("🔍 Auto-discovering Edge bookmark files...")
        
        # Common Edge profile locations
        potential_paths = []
        
        # Windows paths
        import os
        user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        
        windows_paths = [
            Path.home() / "AppData/Local/Microsoft/Edge/User Data",
            Path(f"/c/Users/{user}/AppData/Local/Microsoft/Edge/User Data"),
            Path(f"/mnt/c/Users/{user}/AppData/Local/Microsoft/Edge/User Data"),
        ]
        
        for base_path in windows_paths:
            if base_path.exists():
                # Check default profile
                default_bookmark = base_path / "Default/Bookmarks"
                if default_bookmark.exists():
                    potential_paths.append(default_bookmark)
                
                # Check additional profiles
                for profile_dir in base_path.glob("Profile *"):
                    if profile_dir.is_dir():
                        profile_bookmark = profile_dir / "Bookmarks"
                        if profile_bookmark.exists():
                            potential_paths.append(profile_bookmark)
                break
        
        self.monitored_paths.update(potential_paths)
        logger.info(f"📁 Discovered {len(potential_paths)} bookmark files")
    
    async def _start_watchdog_monitoring(self) -> None:
        """Start watchdog-based monitoring."""
        try:
            self.observer = Observer()
            handler = BookmarkFileHandler(
                self._handle_file_change,
                config.file_monitor.debounce_seconds
            )
            
            # Watch each profile directory
            watched_dirs = set()
            for bookmark_path in self.monitored_paths:
                profile_dir = bookmark_path.parent
                if profile_dir not in watched_dirs:
                    self.observer.schedule(handler, str(profile_dir), recursive=False)
                    watched_dirs.add(profile_dir)
                    logger.debug(f"👁️ Watching directory: {profile_dir}")
            
            self.observer.start()
            self.stats['monitoring_method'] = 'watchdog'
            logger.info("✅ Watchdog monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Watchdog monitoring failed: {e}")
            self.stats['error_count'] += 1
            # Fall back to polling
            await self._start_polling_monitoring()
    
    async def _start_polling_monitoring(self) -> None:
        """Start polling-based monitoring."""
        try:
            self.polling_monitor = PollingMonitor(
                list(self.monitored_paths),
                self._handle_file_change,
                poll_interval=config.file_monitor.debounce_seconds * 2
            )
            
            await self.polling_monitor.start()
            self.stats['monitoring_method'] = 'polling'
            logger.info("✅ Polling monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Polling monitoring failed: {e}")
            self.stats['error_count'] += 1
    
    async def _handle_file_change(self, file_path: str, event_type: str) -> None:
        """Handle bookmark file changes."""
        try:
            self.stats['events_processed'] += 1
            self.stats['last_event_time'] = datetime.now()
            
            logger.info(f"📝 Bookmark file {event_type}: {file_path}")
            
            # Notify subscribers
            await self._notify_subscribers(file_path, event_type)
            
            # Call main callback if provided
            if self.on_change_callback:
                await self.on_change_callback(file_path, event_type)
                
        except Exception as e:
            logger.error(f"❌ Error handling file change: {e}")
            self.stats['error_count'] += 1
    
    async def _notify_subscribers(self, file_path: str, event_type: str) -> None:
        """Notify all subscribers of file changes."""
        if not self.subscribers:
            return
        
        notification = {
            'event': 'bookmark_file_changed',
            'file_path': file_path,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Notify all subscribers
        failed_subscribers = []
        for session_id, subscriber_info in self.subscribers.items():
            try:
                callback = subscriber_info.get('callback')
                if callback:
                    await callback(notification)
            except Exception as e:
                logger.warning(f"⚠️ Failed to notify subscriber {session_id}: {e}")
                failed_subscribers.append(session_id)
        
        # Clean up failed subscribers
        for session_id in failed_subscribers:
            self.remove_subscriber(session_id)
    
    def add_subscriber(self, session_id: str, callback: Optional[Callable] = None) -> None:
        """Add a subscriber for file change notifications."""
        self.subscribers[session_id] = {
            'callback': callback,
            'subscribed_at': datetime.now(),
            'notifications_sent': 0
        }
        logger.info(f"➕ Added subscriber: {session_id}")
    
    def remove_subscriber(self, session_id: str) -> None:
        """Remove a subscriber."""
        if session_id in self.subscribers:
            del self.subscribers[session_id]
            logger.info(f"➖ Removed subscriber: {session_id}")
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'is_running': self.is_running,
            'monitoring_method': self.stats['monitoring_method'],
            'monitored_files': [str(path) for path in self.monitored_paths],
            'subscriber_count': len(self.subscribers),
            'statistics': {
                'events_processed': self.stats['events_processed'],
                'error_count': self.stats['error_count'],
                'last_event_time': self.stats['last_event_time'].isoformat() if self.stats['last_event_time'] else None,
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'uptime_seconds': (
                    (datetime.now() - self.stats['start_time']).total_seconds() 
                    if self.stats['start_time'] else 0
                )
            },
            'capabilities': {
                'watchdog_available': WATCHDOG_AVAILABLE,
                'real_time_monitoring': WATCHDOG_AVAILABLE,
                'debouncing': True,
                'multi_profile_support': True
            }
        }
    
    async def validate_monitored_files(self) -> Dict[str, bool]:
        """Validate that all monitored files still exist."""
        validation_results = {}
        
        for path in self.monitored_paths:
            validation_results[str(path)] = path.exists()
        
        missing_files = [path for path, exists in validation_results.items() if not exists]
        if missing_files:
            logger.warning(f"⚠️ Missing bookmark files: {missing_files}")
        
        return validation_results
    
    async def refresh_monitored_paths(self) -> int:
        """Refresh the list of monitored paths."""
        logger.info("🔄 Refreshing monitored bookmark paths...")
        
        old_count = len(self.monitored_paths)
        
        # Re-discover bookmark files
        self.monitored_paths.clear()
        await self._discover_bookmark_files()
        
        new_count = len(self.monitored_paths)
        logger.info(f"📊 Path refresh: {old_count} -> {new_count} files")
        
        # Restart monitoring if running
        if self.is_running:
            await self.stop_monitoring()
            await self.start_monitoring()
        
        return new_count
    
    def get_subscriber_info(self) -> Dict[str, Dict]:
        """Get information about current subscribers."""
        return {
            session_id: {
                'subscribed_at': info['subscribed_at'].isoformat(),
                'notifications_sent': info['notifications_sent'],
                'has_callback': info['callback'] is not None
            }
            for session_id, info in self.subscribers.items()
        }