"""
Configuration module for Edge Bookmark MCP Server
Handles server settings, environment variables, and feature flags.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    """Available search modes."""
    FUZZY = "fuzzy"
    EXACT = "exact"
    SEMANTIC = "semantic"

class LogLevel(Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SearchConfig:
    """Search engine configuration."""
    default_mode: SearchMode = SearchMode.FUZZY
    default_limit: int = 20
    max_limit: int = 100
    min_score_threshold: float = 60.0
    fuzzy_ratio_weight: float = 0.4
    partial_ratio_weight: float = 0.2
    token_set_ratio_weight: float = 0.3
    tag_score_weight: float = 0.1
    enable_caching: bool = True
    cache_size: int = 1000
    suggestion_limit: int = 10

@dataclass
class FileMonitorConfig:
    """File monitoring configuration."""
    enabled: bool = True
    debounce_seconds: float = 2.0
    watch_recursive: bool = True
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "**/tmp/*", "**/.tmp*", "**/lock*", "**/~*"
    ])

@dataclass
class BackupConfig:
    """Backup configuration."""
    enabled: bool = True
    max_backups_per_profile: int = 10
    backup_directory: str = "data/backups"
    auto_cleanup: bool = True
    backup_before_modification: bool = True

@dataclass
class ExportConfig:
    """Export functionality configuration."""
    supported_formats: List[str] = field(default_factory=lambda: [
        "json", "csv", "html", "xlsx"
    ])
    default_format: str = "json"
    include_metadata_by_default: bool = True
    max_export_size: int = 100000  # Maximum bookmarks per export
    output_directory: str = "data/exports"

@dataclass
class AnalyticsConfig:
    """Analytics configuration."""
    duplicate_similarity_threshold: float = 80.0
    domain_analysis_enabled: bool = True
    folder_analysis_enabled: bool = True
    health_scoring_enabled: bool = True
    recommendation_engine_enabled: bool = True
    max_folder_size_warning: int = 50

@dataclass
class SecurityConfig:
    """Security and safety configuration."""
    validate_urls: bool = True
    sanitize_inputs: bool = True
    max_query_length: int = 500
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".json", ".csv", ".html", ".xlsx"
    ])
    block_suspicious_urls: bool = True
    rate_limit_enabled: bool = False
    max_requests_per_minute: int = 100

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    async_operations: bool = True
    max_concurrent_operations: int = 10
    chunk_size_large_datasets: int = 1000
    enable_compression: bool = True
    memory_cache_size_mb: int = 100
    disk_cache_enabled: bool = True
    disk_cache_directory: str = "data/cache"

class ServerConfig:
    """Main server configuration class."""
    
    def __init__(self):
        self.search = SearchConfig()
        self.file_monitor = FileMonitorConfig()
        self.backup = BackupConfig()
        self.export = ExportConfig()
        self.analytics = AnalyticsConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Server settings
        self.server_name = "edge-bookmark-server"
        self.server_version = "1.0.0"
        self.server_description = "Microsoft Edge bookmark management with advanced search and analytics"
        
        # Logging configuration
        self.log_level = LogLevel.INFO
        self.log_file = "edge-bookmark-server.log"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_max_bytes = 10 * 1024 * 1024  # 10MB
        self.log_backup_count = 5
        
        # FastMCP specific settings
        self.mcp_port = None  # Let FastMCP choose
        self.mcp_host = "localhost"
        self.mcp_debug = False
        
        # Data directories
        self.data_directory = Path("data")
        self.ensure_directories()
        
        # Load configuration from environment
        self.load_from_environment()
        
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_directory,
            Path(self.backup.backup_directory),
            Path(self.export.output_directory),
            Path(self.performance.disk_cache_directory)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def load_from_environment(self):
        """Load configuration from environment variables."""
        
        # Search configuration
        if os.getenv("BOOKMARK_SEARCH_MODE"):
            try:
                self.search.default_mode = SearchMode(os.getenv("BOOKMARK_SEARCH_MODE"))
            except ValueError:
                logger.warning(f"Invalid search mode: {os.getenv('BOOKMARK_SEARCH_MODE')}")
        
        if os.getenv("BOOKMARK_DEFAULT_LIMIT"):
            try:
                self.search.default_limit = int(os.getenv("BOOKMARK_DEFAULT_LIMIT"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_DEFAULT_LIMIT value")
        
        if os.getenv("BOOKMARK_SCORE_THRESHOLD"):
            try:
                self.search.min_score_threshold = float(os.getenv("BOOKMARK_SCORE_THRESHOLD"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_SCORE_THRESHOLD value")
        
        # File monitoring
        if os.getenv("BOOKMARK_MONITOR_ENABLED"):
            self.file_monitor.enabled = os.getenv("BOOKMARK_MONITOR_ENABLED").lower() == "true"
        
        if os.getenv("BOOKMARK_DEBOUNCE_SECONDS"):
            try:
                self.file_monitor.debounce_seconds = float(os.getenv("BOOKMARK_DEBOUNCE_SECONDS"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_DEBOUNCE_SECONDS value")
        
        # Backup configuration
        if os.getenv("BOOKMARK_BACKUP_ENABLED"):
            self.backup.enabled = os.getenv("BOOKMARK_BACKUP_ENABLED").lower() == "true"
        
        if os.getenv("BOOKMARK_BACKUP_DIR"):
            self.backup.backup_directory = os.getenv("BOOKMARK_BACKUP_DIR")
        
        if os.getenv("BOOKMARK_MAX_BACKUPS"):
            try:
                self.backup.max_backups_per_profile = int(os.getenv("BOOKMARK_MAX_BACKUPS"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_MAX_BACKUPS value")
        
        # Export configuration
        if os.getenv("BOOKMARK_EXPORT_DIR"):
            self.export.output_directory = os.getenv("BOOKMARK_EXPORT_DIR")
        
        if os.getenv("BOOKMARK_DEFAULT_EXPORT_FORMAT"):
            format_val = os.getenv("BOOKMARK_DEFAULT_EXPORT_FORMAT")
            if format_val in self.export.supported_formats:
                self.export.default_format = format_val
            else:
                logger.warning(f"Invalid export format: {format_val}")
        
        # Analytics configuration
        if os.getenv("BOOKMARK_DUPLICATE_THRESHOLD"):
            try:
                self.analytics.duplicate_similarity_threshold = float(os.getenv("BOOKMARK_DUPLICATE_THRESHOLD"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_DUPLICATE_THRESHOLD value")
        
        # Security configuration
        if os.getenv("BOOKMARK_VALIDATE_URLS"):
            self.security.validate_urls = os.getenv("BOOKMARK_VALIDATE_URLS").lower() == "true"
        
        if os.getenv("BOOKMARK_MAX_QUERY_LENGTH"):
            try:
                self.security.max_query_length = int(os.getenv("BOOKMARK_MAX_QUERY_LENGTH"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_MAX_QUERY_LENGTH value")
        
        # Performance configuration
        if os.getenv("BOOKMARK_ASYNC_OPERATIONS"):
            self.performance.async_operations = os.getenv("BOOKMARK_ASYNC_OPERATIONS").lower() == "true"
        
        if os.getenv("BOOKMARK_MAX_CONCURRENT"):
            try:
                self.performance.max_concurrent_operations = int(os.getenv("BOOKMARK_MAX_CONCURRENT"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_MAX_CONCURRENT value")
        
        if os.getenv("BOOKMARK_CACHE_SIZE_MB"):
            try:
                self.performance.memory_cache_size_mb = int(os.getenv("BOOKMARK_CACHE_SIZE_MB"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_CACHE_SIZE_MB value")
        
        # Logging configuration
        if os.getenv("BOOKMARK_LOG_LEVEL"):
            try:
                self.log_level = LogLevel(os.getenv("BOOKMARK_LOG_LEVEL").upper())
            except ValueError:
                logger.warning(f"Invalid log level: {os.getenv('BOOKMARK_LOG_LEVEL')}")
        
        if os.getenv("BOOKMARK_LOG_FILE"):
            self.log_file = os.getenv("BOOKMARK_LOG_FILE")
        
        # FastMCP configuration
        if os.getenv("BOOKMARK_MCP_PORT"):
            try:
                self.mcp_port = int(os.getenv("BOOKMARK_MCP_PORT"))
            except ValueError:
                logger.warning("Invalid BOOKMARK_MCP_PORT value")
        
        if os.getenv("BOOKMARK_MCP_HOST"):
            self.mcp_host = os.getenv("BOOKMARK_MCP_HOST")
        
        if os.getenv("BOOKMARK_DEBUG_MODE"):
            self.mcp_debug = os.getenv("BOOKMARK_DEBUG_MODE").lower() == "true"
        
        # Data directory
        if os.getenv("BOOKMARK_DATA_DIR"):
            self.data_directory = Path(os.getenv("BOOKMARK_DATA_DIR"))
            self.ensure_directories()
        
        logger.info("Configuration loaded from environment variables")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "server": {
                "name": self.server_name,
                "version": self.server_version,
                "description": self.server_description,
                "log_level": self.log_level.value,
                "log_file": self.log_file,
                "data_directory": str(self.data_directory)
            },
            "search": {
                "default_mode": self.search.default_mode.value,
                "default_limit": self.search.default_limit,
                "max_limit": self.search.max_limit,
                "min_score_threshold": self.search.min_score_threshold,
                "enable_caching": self.search.enable_caching,
                "cache_size": self.search.cache_size
            },
            "file_monitor": {
                "enabled": self.file_monitor.enabled,
                "debounce_seconds": self.file_monitor.debounce_seconds,
                "watch_recursive": self.file_monitor.watch_recursive
            },
            "backup": {
                "enabled": self.backup.enabled,
                "max_backups_per_profile": self.backup.max_backups_per_profile,
                "backup_directory": self.backup.backup_directory,
                "auto_cleanup": self.backup.auto_cleanup
            },
            "export": {
                "supported_formats": self.export.supported_formats,
                "default_format": self.export.default_format,
                "include_metadata_by_default": self.export.include_metadata_by_default,
                "output_directory": self.export.output_directory
            },
            "analytics": {
                "duplicate_similarity_threshold": self.analytics.duplicate_similarity_threshold,
                "domain_analysis_enabled": self.analytics.domain_analysis_enabled,
                "folder_analysis_enabled": self.analytics.folder_analysis_enabled,
                "health_scoring_enabled": self.analytics.health_scoring_enabled
            },
            "security": {
                "validate_urls": self.security.validate_urls,
                "sanitize_inputs": self.security.sanitize_inputs,
                "max_query_length": self.security.max_query_length,
                "block_suspicious_urls": self.security.block_suspicious_urls
            },
            "performance": {
                "async_operations": self.performance.async_operations,
                "max_concurrent_operations": self.performance.max_concurrent_operations,
                "chunk_size_large_datasets": self.performance.chunk_size_large_datasets,
                "memory_cache_size_mb": self.performance.memory_cache_size_mb,
                "disk_cache_enabled": self.performance.disk_cache_enabled
            },
            "mcp": {
                "port": self.mcp_port,
                "host": self.mcp_host,
                "debug": self.mcp_debug
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate search configuration
        if self.search.default_limit <= 0:
            issues.append("Search default_limit must be positive")
        
        if self.search.max_limit < self.search.default_limit:
            issues.append("Search max_limit must be >= default_limit")
        
        if not (0 <= self.search.min_score_threshold <= 100):
            issues.append("Search min_score_threshold must be between 0 and 100")
        
        # Validate weight sum for fuzzy scoring
        weight_sum = (
            self.search.fuzzy_ratio_weight +
            self.search.partial_ratio_weight +
            self.search.token_set_ratio_weight +
            self.search.tag_score_weight
        )
        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
            issues.append(f"Search weights must sum to 1.0, got {weight_sum}")
        
        # Validate file monitor configuration
        if self.file_monitor.debounce_seconds < 0:
            issues.append("File monitor debounce_seconds must be non-negative")
        
        # Validate backup configuration
        if self.backup.max_backups_per_profile <= 0:
            issues.append("Backup max_backups_per_profile must be positive")
        
        # Validate export configuration
        if self.export.default_format not in self.export.supported_formats:
            issues.append(f"Export default_format '{self.export.default_format}' not in supported_formats")
        
        if self.export.max_export_size <= 0:
            issues.append("Export max_export_size must be positive")
        
        # Validate analytics configuration
        if not (0 <= self.analytics.duplicate_similarity_threshold <= 100):
            issues.append("Analytics duplicate_similarity_threshold must be between 0 and 100")
        
        if self.analytics.max_folder_size_warning <= 0:
            issues.append("Analytics max_folder_size_warning must be positive")
        
        # Validate security configuration
        if self.security.max_query_length <= 0:
            issues.append("Security max_query_length must be positive")
        
        if self.security.rate_limit_enabled and self.security.max_requests_per_minute <= 0:
            issues.append("Security max_requests_per_minute must be positive when rate limiting enabled")
        
        # Validate performance configuration
        if self.performance.max_concurrent_operations <= 0:
            issues.append("Performance max_concurrent_operations must be positive")
        
        if self.performance.chunk_size_large_datasets <= 0:
            issues.append("Performance chunk_size_large_datasets must be positive")
        
        if self.performance.memory_cache_size_mb <= 0:
            issues.append("Performance memory_cache_size_mb must be positive")
        
        # Validate directories exist
        try:
            if not self.data_directory.exists():
                issues.append(f"Data directory does not exist: {self.data_directory}")
        except Exception as e:
            issues.append(f"Cannot access data directory: {e}")
        
        # Validate MCP configuration
        if self.mcp_port is not None and not (1 <= self.mcp_port <= 65535):
            issues.append("MCP port must be between 1 and 65535")
        
        return issues
    
    def get_environment_template(self) -> str:
        """Generate environment variable template for configuration."""
        template = """# Edge Bookmark MCP Server Configuration
# Copy this file to .env and modify as needed

# Search Configuration
BOOKMARK_SEARCH_MODE=fuzzy  # fuzzy, exact, semantic
BOOKMARK_DEFAULT_LIMIT=20
BOOKMARK_SCORE_THRESHOLD=60.0

# File Monitoring
BOOKMARK_MONITOR_ENABLED=true
BOOKMARK_DEBOUNCE_SECONDS=2.0

# Backup Configuration  
BOOKMARK_BACKUP_ENABLED=true
BOOKMARK_BACKUP_DIR=data/backups
BOOKMARK_MAX_BACKUPS=10

# Export Configuration
BOOKMARK_EXPORT_DIR=data/exports
BOOKMARK_DEFAULT_EXPORT_FORMAT=json

# Analytics Configuration
BOOKMARK_DUPLICATE_THRESHOLD=80.0

# Security Configuration
BOOKMARK_VALIDATE_URLS=true
BOOKMARK_MAX_QUERY_LENGTH=500

# Performance Configuration
BOOKMARK_ASYNC_OPERATIONS=true
BOOKMARK_MAX_CONCURRENT=10
BOOKMARK_CACHE_SIZE_MB=100

# Logging Configuration
BOOKMARK_LOG_LEVEL=INFO
BOOKMARK_LOG_FILE=edge-bookmark-server.log

# FastMCP Configuration
BOOKMARK_MCP_PORT=8080
BOOKMARK_MCP_HOST=localhost
BOOKMARK_DEBUG_MODE=false

# Data Directory
BOOKMARK_DATA_DIR=data
"""
        return template.strip()
    
    def save_template_file(self, file_path: str = ".env.template") -> None:
        """Save environment template to file."""
        template = self.get_environment_template()
        with open(file_path, 'w') as f:
            f.write(template)
        logger.info(f"Environment template saved to {file_path}")
    
    def apply_runtime_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply runtime configuration overrides."""
        for key, value in overrides.items():
            parts = key.split('.')
            
            if len(parts) == 2:
                section, setting = parts
                
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, setting):
                        old_value = getattr(section_obj, setting)
                        setattr(section_obj, setting, value)
                        logger.info(f"Applied runtime override: {key} = {value} (was {old_value})")
                    else:
                        logger.warning(f"Unknown setting in runtime override: {key}")
                else:
                    logger.warning(f"Unknown section in runtime override: {section}")
            else:
                logger.warning(f"Invalid runtime override key format: {key}")
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get current feature flag status."""
        return {
            "file_monitoring": self.file_monitor.enabled,
            "backup_creation": self.backup.enabled,
            "auto_backup_cleanup": self.backup.auto_cleanup,
            "domain_analysis": self.analytics.domain_analysis_enabled,
            "folder_analysis": self.analytics.folder_analysis_enabled,
            "health_scoring": self.analytics.health_scoring_enabled,
            "recommendation_engine": self.analytics.recommendation_engine_enabled,
            "url_validation": self.security.validate_urls,
            "input_sanitization": self.security.sanitize_inputs,
            "suspicious_url_blocking": self.security.block_suspicious_urls,
            "rate_limiting": self.security.rate_limit_enabled,
            "async_operations": self.performance.async_operations,
            "compression": self.performance.enable_compression,
            "disk_cache": self.performance.disk_cache_enabled,
            "search_caching": self.search.enable_caching
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings summary."""
        return {
            "async_operations": self.performance.async_operations,
            "max_concurrent_operations": self.performance.max_concurrent_operations,
            "chunk_size_large_datasets": self.performance.chunk_size_large_datasets,
            "memory_cache_size_mb": self.performance.memory_cache_size_mb,
            "disk_cache_enabled": self.performance.disk_cache_enabled,
            "search_cache_size": self.search.cache_size,
            "search_caching_enabled": self.search.enable_caching,
            "compression_enabled": self.performance.enable_compression
        }

# Global configuration instance
config = ServerConfig()