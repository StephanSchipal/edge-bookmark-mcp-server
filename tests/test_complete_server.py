#!/usr/bin/env python3
"""
Comprehensive Test Suite for Edge Bookmark MCP Server
Tests all components, integration scenarios, and error handling
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEdgeBookmarkServer(unittest.TestCase):
    """Comprehensive test suite for the Edge Bookmark MCP Server."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_bookmarks = [
            {
                'id': '1',
                'title': 'GitHub',
                'url': 'https://github.com',
                'folder_path': 'Development',
                'date_added': '2025-01-01T00:00:00Z',
                'tags': ['code', 'development'],
                'profile': 'Default',
                'description': 'Code repository platform'
            },
            {
                'id': '2',
                'title': 'Stack Overflow',
                'url': 'https://stackoverflow.com',
                'folder_path': 'Development/Help',
                'date_added': '2025-01-02T00:00:00Z',
                'tags': ['programming', 'help'],
                'profile': 'Default',
                'description': 'Programming Q&A site'
            },
            {
                'id': '3',
                'title': 'Google',
                'url': 'https://google.com',
                'folder_path': 'Search',
                'date_added': '2025-01-03T00:00:00Z',
                'tags': ['search', 'web'],
                'profile': 'Default',
                'description': 'Search engine'
            },
            {
                'id': '4',
                'title': 'GitHub Enterprise',
                'url': 'https://github.enterprise.com',
                'folder_path': 'Development',
                'date_added': '2025-01-04T00:00:00Z',
                'tags': ['code', 'enterprise'],
                'profile': 'Work',
                'description': 'Enterprise GitHub'
            }
        ]
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        logger.info(f"Test setup completed with {len(self.test_bookmarks)} mock bookmarks")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_configuration_system(self):
        """Test configuration loading and validation."""
        logger.info("🔧 Testing configuration system...")
        
        try:
            from config import ServerConfig
            
            # Test basic configuration
            config = ServerConfig()
            self.assertIsNotNone(config.server_name)
            self.assertIsNotNone(config.server_version)
            
            # Test validation
            issues = config.validate_configuration()
            self.assertIsInstance(issues, list)
            
            # Test serialization
            config_dict = config.to_dict()
            self.assertIn('server', config_dict)
            self.assertIn('search', config_dict)
            
            logger.info("✅ Configuration system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    async def test_bookmark_loader(self):
        """Test bookmark loading functionality."""
        logger.info("📁 Testing bookmark loader...")
        
        try:
            from bookmark_loader import EdgeBookmarkLoader
            
            loader = EdgeBookmarkLoader()