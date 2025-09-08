"""
Bookmark Search Engine - Advanced fuzzy search with RapidFuzz
Implements multi-field searching, caching, and various search modes.
"""

import asyncio
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urlparse
import re

from rapidfuzz import fuzz, process, utils
from config import config

logger = logging.getLogger(__name__)

class BookmarkSearchEngine:
    """Advanced search engine for bookmark collections."""
    
    def __init__(self):
        self.bookmarks: List[Dict[str, Any]] = []
        self.search_index: Dict[str, Any] = {}
        self.fuzzy_cache: Dict[str, Dict[str, Any]] = {}
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.performance.max_concurrent_operations)
        self.last_index_time: Optional[float] = None
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'average_search_time_ms': 0.0,
            'most_searched_terms': defaultdict(int)
        }
    
    async def index_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Index bookmarks for fast searching."""
        start_time = time.time()
        logger.info(f"🔍 Indexing {len(bookmarks)} bookmarks...")
        
        self.bookmarks = bookmarks
        self.fuzzy_cache.clear()
        self.query_cache.clear()
        
        # Build search index in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self._build_search_index
        )
        
        index_time = (time.time() - start_time) * 1000
        self.last_index_time = time.time()
        
        logger.info(f"✅ Search index built in {index_time:.1f}ms for {len(bookmarks)} bookmarks")
    
    def _build_search_index(self) -> None:
        """Build search index structures (runs in thread pool)."""
        self.search_index = {
            'by_title': {},
            'by_url': {},
            'by_domain': {},
            'by_folder': {},
            'by_tags': {},
            'word_index': defaultdict(set)
        }
        
        for i, bookmark in enumerate(self.bookmarks):
            bookmark_id = bookmark.get('id', str(i))
            
            # Index by title
            title = bookmark.get('title', '').lower()
            if title:
                self.search_index['by_title'][bookmark_id] = title
                # Index individual words
                for word in re.findall(r'\b\w+\b', title):
                    if len(word) >= 2:  # Skip single characters
                        self.search_index['word_index'][word].add(bookmark_id)
            
            # Index by URL
            url = bookmark.get('url', '').lower()
            if url:
                self.search_index['by_url'][bookmark_id] = url
                
                # Extract and index domain
                try:
                    domain = urlparse(url).netloc
                    if domain:
                        self.search_index['by_domain'][bookmark_id] = domain
                        # Index domain words
                        for part in domain.split('.'):
                            if len(part) >= 2:
                                self.search_index['word_index'][part].add(bookmark_id)
                except:
                    pass
            
            # Index by folder
            folder = bookmark.get('folder_path', '').lower()
            if folder:
                self.search_index['by_folder'][bookmark_id] = folder
                # Index folder parts
                for part in folder.split('/'):
                    if len(part) >= 2:
                        self.search_index['word_index'][part].add(bookmark_id)
            
            # Index by tags
            tags = bookmark.get('tags', [])
            if tags:
                self.search_index['by_tags'][bookmark_id] = [tag.lower() for tag in tags]
                for tag in tags:
                    if len(tag) >= 2:
                        self.search_index['word_index'][tag.lower()].add(bookmark_id)
            
            # Preprocess for fuzzy matching
            self.fuzzy_cache[bookmark_id] = {
                'title_processed': utils.default_process(title) if title else '',
                'url_processed': url,
                'folder_processed': utils.default_process(folder) if folder else '',
                'tags_processed': [utils.default_process(tag) for tag in tags if tag],
                'description_processed': utils.default_process(
                    bookmark.get('description', '')
                ) if bookmark.get('description') else ''
            }
    
    async def fuzzy_search(
        self, 
        query: str, 
        limit: int = None, 
        folder_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform fuzzy search with multi-field scoring."""
        if limit is None:
            limit = config.search.default_limit
        
        limit = min(limit, config.search.max_limit)
        
        # Check cache first
        cache_key = f"fuzzy:{query}:{limit}:{folder_filter or 'all'}"
        if config.search.enable_caching and cache_key in self.query_cache:
            self.search_stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        start_time = time.time()
        processed_query = utils.default_process(query)
        
        # Run search in thread pool
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor, self._fuzzy_search_sync, processed_query, query, limit, folder_filter
        )
        
        search_time = (time.time() - start_time) * 1000
        self._update_search_stats(query, search_time)
        
        # Cache results
        if config.search.enable_caching:
            self._cache_results(cache_key, results)
        
        return results
    
    def _fuzzy_search_sync(
        self, 
        processed_query: str, 
        original_query: str,
        limit: int, 
        folder_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Synchronous fuzzy search implementation."""
        scored_results = []
        
        for i, bookmark in enumerate(self.bookmarks):
            bookmark_id = bookmark.get('id', str(i))
            
            # Apply folder filter if specified
            if folder_filter:
                folder_path = bookmark.get('folder_path', '')
                if not folder_path.lower().startswith(folder_filter.lower()):
                    continue
            
            cached_data = self.fuzzy_cache.get(bookmark_id, {})
            
            # Multi-field fuzzy scoring
            scores = self._calculate_fuzzy_scores(
                processed_query, original_query, cached_data, bookmark
            )
            
            # Calculate composite score using configured weights
            composite_score = (
                scores['title'] * config.search.fuzzy_ratio_weight +
                scores['url'] * config.search.partial_ratio_weight +
                scores['description'] * config.search.token_set_ratio_weight +
                scores['tags'] * config.search.tag_score_weight
            )
            
            # Apply threshold filter
            if composite_score >= config.search.min_score_threshold:
                scored_results.append({
                    'bookmark': bookmark,
                    'score': composite_score,
                    'match_breakdown': scores,
                    'search_query': original_query
                })
        
        # Sort by score and limit results
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:limit]
    
    def _calculate_fuzzy_scores(
        self, 
        processed_query: str, 
        original_query: str,
        cached_data: Dict[str, Any], 
        bookmark: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate fuzzy match scores for all fields."""
        scores = {
            'title': 0.0,
            'url': 0.0,
            'description': 0.0,
            'tags': 0.0,
            'folder': 0.0
        }
        
        # Title scoring (highest weight)
        title_processed = cached_data.get('title_processed', '')
        if title_processed:
            scores['title'] = fuzz.WRatio(processed_query, title_processed)
        
        # URL scoring (partial matches for domains)
        url_processed = cached_data.get('url_processed', '')
        if url_processed:
            scores['url'] = fuzz.partial_ratio(original_query.lower(), url_processed)
        
        # Description scoring (token-based for better phrase matching)
        desc_processed = cached_data.get('description_processed', '')
        if desc_processed:
            scores['description'] = fuzz.token_set_ratio(processed_query, desc_processed)
        
        # Tag scoring (best matching tag)
        tags_processed = cached_data.get('tags_processed', [])
        if tags_processed:
            tag_scores = [fuzz.ratio(processed_query, tag) for tag in tags_processed]
            scores['tags'] = max(tag_scores) if tag_scores else 0
        
        # Folder scoring
        folder_processed = cached_data.get('folder_processed', '')
        if folder_processed:
            scores['folder'] = fuzz.partial_ratio(processed_query, folder_processed)
        
        return scores
    
    async def exact_search(
        self, 
        query: str, 
        limit: int = None, 
        folder_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform exact string matching search."""
        if limit is None:
            limit = config.search.default_limit
        
        limit = min(limit, config.search.max_limit)
        
        # Check cache
        cache_key = f"exact:{query}:{limit}:{folder_filter or 'all'}"
        if config.search.enable_caching and cache_key in self.query_cache:
            self.search_stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        start_time = time.time()
        query_lower = query.lower()
        
        results = []
        for bookmark in self.bookmarks:
            # Apply folder filter
            if folder_filter:
                folder_path = bookmark.get('folder_path', '')
                if not folder_path.lower().startswith(folder_filter.lower()):
                    continue
            
            score = 0.0
            matches = []
            
            # Check title
            title = bookmark.get('title', '').lower()
            if query_lower in title:
                score += 100 if query_lower == title else 80
                matches.append('title')
            
            # Check URL
            url = bookmark.get('url', '').lower()
            if query_lower in url:
                score += 90 if query_lower in url else 60
                matches.append('url')
            
            # Check tags
            for tag in bookmark.get('tags', []):
                if query_lower in tag.lower():
                    score += 70
                    matches.append('tags')
                    break
            
            # Check folder
            folder = bookmark.get('folder_path', '').lower()
            if query_lower in folder:
                score += 50
                matches.append('folder')
            
            if score > 0:
                results.append({
                    'bookmark': bookmark,
                    'score': score,
                    'match_breakdown': {
                        'matches': matches,
                        'exact_match': query_lower in title or query_lower in url
                    },
                    'search_query': query
                })
        
        # Sort and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:limit]
        
        search_time = (time.time() - start_time) * 1000
        self._update_search_stats(query, search_time)
        
        # Cache results
        if config.search.enable_caching:
            self._cache_results(cache_key, results)
        
        return results
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = None, 
        folder_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using word similarity and context."""
        if limit is None:
            limit = config.search.default_limit
        
        limit = min(limit, config.search.max_limit)
        
        # For now, implement as enhanced fuzzy search with word expansion
        # In a full implementation, this would use embeddings/NLP
        
        start_time = time.time()
        
        # Expand query with related terms
        expanded_query = await self._expand_query_terms(query)
        
        # Use fuzzy search with expanded terms
        results = await self.fuzzy_search(expanded_query, limit * 2, folder_filter)
        
        # Re-rank results based on semantic relevance
        semantic_results = []
        for result in results:
            semantic_score = self._calculate_semantic_score(query, result['bookmark'])
            
            semantic_results.append({
                'bookmark': result['bookmark'],
                'score': semantic_score,
                'match_breakdown': {
                    'fuzzy_score': result['score'],
                    'semantic_score': semantic_score,
                    'expanded_query': expanded_query
                },
                'search_query': query
            })
        
        # Sort by semantic score and limit
        semantic_results.sort(key=lambda x: x['score'], reverse=True)
        semantic_results = semantic_results[:limit]
        
        search_time = (time.time() - start_time) * 1000
        self._update_search_stats(f"semantic:{query}", search_time)
        
        return semantic_results
    
    async def _expand_query_terms(self, query: str) -> str:
        """Expand query with related terms based on bookmark content."""
        # Simple term expansion based on common bookmark patterns
        expansions = {
            'dev': ['development', 'developer', 'programming'],
            'js': ['javascript', 'node', 'react'],
            'py': ['python', 'django', 'flask'],
            'doc': ['documentation', 'docs', 'manual'],
            'api': ['rest', 'endpoint', 'service'],
            'db': ['database', 'sql', 'mysql', 'postgres'],
            'ui': ['interface', 'design', 'frontend'],
            'ml': ['machine learning', 'ai', 'data science']
        }
        
        query_words = query.lower().split()
        expanded_terms = list(query_words)
        
        for word in query_words:
            if word in expansions:
                expanded_terms.extend(expansions[word])
        
        return ' '.join(expanded_terms)
    
    def _calculate_semantic_score(self, query: str, bookmark: Dict[str, Any]) -> float:
        """Calculate semantic relevance score."""
        # Simple semantic scoring based on word co-occurrence and context
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Extract all text from bookmark
        text_fields = [
            bookmark.get('title', ''),
            bookmark.get('url', ''),
            bookmark.get('description', ''),
            ' '.join(bookmark.get('tags', [])),
            bookmark.get('folder_path', '')
        ]
        
        all_text = ' '.join(text_fields).lower()
        bookmark_words = set(re.findall(r'\b\w+\b', all_text))
        
        # Calculate word overlap
        overlap = len(query_words.intersection(bookmark_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        # Basic semantic score
        base_score = (overlap / total_query_words) * 100
        
        # Boost for title matches
        title_words = set(re.findall(r'\b\w+\b', bookmark.get('title', '').lower()))
        title_overlap = len(query_words.intersection(title_words))
        title_boost = (title_overlap / total_query_words) * 20
        
        return min(100.0, base_score + title_boost)
    
    async def get_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query."""
        if len(partial_query) < 2:
            return []
        
        suggestions = set()
        partial_lower = partial_query.lower()
        
        # Search in word index
        for word, bookmark_ids in self.search_index.get('word_index', {}).items():
            if word.startswith(partial_lower) and len(word) > len(partial_lower):
                suggestions.add(word)
        
        # Search in titles and tags from recent popular terms
        for term, count in self.search_stats['most_searched_terms'].items():
            if term.lower().startswith(partial_lower) and len(term) > len(partial_query):
                suggestions.add(term)
        
        # Sort by relevance (length and frequency)
        suggestion_list = list(suggestions)
        suggestion_list.sort(key=lambda x: (len(x), -self.search_stats['most_searched_terms'].get(x, 0)))
        
        return suggestion_list[:limit]
    
    async def search_by_domain(self, domain: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search bookmarks by domain name."""
        if limit is None:
            limit = config.search.default_limit
        
        domain_lower = domain.lower()
        results = []
        
        for bookmark in self.bookmarks:
            url = bookmark.get('url', '')
            try:
                bookmark_domain = urlparse(url).netloc.lower()
                if domain_lower in bookmark_domain:
                    score = 100 if bookmark_domain == domain_lower else 80
                    results.append({
                        'bookmark': bookmark,
                        'score': score,
                        'match_breakdown': {'domain_match': bookmark_domain},
                        'search_query': domain
                    })
            except:
                continue
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    async def search_by_tags(self, tags: List[str], match_all: bool = False, limit: int = None) -> List[Dict[str, Any]]:
        """Search bookmarks by tags."""
        if limit is None:
            limit = config.search.default_limit
        
        tags_lower = [tag.lower() for tag in tags]
        results = []
        
        for bookmark in self.bookmarks:
            bookmark_tags = [tag.lower() for tag in bookmark.get('tags', [])]
            
            if match_all:
                # All tags must match
                if all(tag in bookmark_tags for tag in tags_lower):
                    score = 100
                    results.append({
                        'bookmark': bookmark,
                        'score': score,
                        'match_breakdown': {'matched_tags': tags_lower, 'match_type': 'all'},
                        'search_query': f"tags:{','.join(tags)}"
                    })
            else:
                # Any tag can match
                matched_tags = [tag for tag in tags_lower if tag in bookmark_tags]
                if matched_tags:
                    score = (len(matched_tags) / len(tags_lower)) * 100
                    results.append({
                        'bookmark': bookmark,
                        'score': score,
                        'match_breakdown': {'matched_tags': matched_tags, 'match_type': 'any'},
                        'search_query': f"tags:{','.join(tags)}"
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _update_search_stats(self, query: str, search_time_ms: float) -> None:
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['most_searched_terms'][query] += 1
        
        # Update average search time
        current_avg = self.search_stats['average_search_time_ms']
        total_searches = self.search_stats['total_searches']
        self.search_stats['average_search_time_ms'] = (
            (current_avg * (total_searches - 1) + search_time_ms) / total_searches
        )
    
    def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results with LRU eviction."""
        if len(self.query_cache) >= config.search.cache_size:
            # Simple LRU: remove oldest entry
            self.query_cache[cache_key] = results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        try:
            import rapidfuzz
            rapidfuzz_available = True
            rapidfuzz_version = rapidfuzz.__version__
        except ImportError:
            rapidfuzz_available = False
            rapidfuzz_version = 'not available'
                
        return {
            'total_bookmarks_indexed': len(self.bookmarks),
            'last_index_time': self.last_index_time,
            'total_searches': self.search_stats['total_searches'],
            'cache_hits': self.search_stats['cache_hits'],
            'cache_hit_ratio': (
                self.search_stats['cache_hits'] / max(1, self.search_stats['total_searches'])
            ) * 100,
            'average_search_time_ms': self.search_stats['average_search_time_ms'],
            'cache_size': len(self.query_cache),
            'rapidfuzz_available': rapidfuzz_available,
            'rapidfuzz_version': rapidfuzz_version,
            'most_searched_terms': dict(sorted(self.search_stats['most_searched_terms'].items(), key=lambda x: x[1], reverse=True)[:10]),
            'index_size_estimate': {
                'word_index_entries': len(self.search_index.get('word_index', {})),
                'fuzzy_cache_entries': len(self.fuzzy_cache),
                'query_cache_entries': len(self.query_cache)
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all search caches."""
        self.query_cache.clear()
        logger.info("Search cache cleared")
    
    def warm_cache(self, common_queries: List[str]) -> None:
        """Warm up cache with common queries."""
        logger.info(f"Warming cache with {len(common_queries)} queries...")
        
        async def warm_query(query: str):
            try:
                await self.fuzzy_search(query, limit=10)
            except Exception as e:
                logger.warning(f"Cache warming failed for query '{query}': {e}")
        
        # Run warming in background
        asyncio.create_task(self._warm_cache_async(common_queries))
    
    async def _warm_cache_async(self, queries: List[str]) -> None:
        """Asynchronously warm cache."""
        tasks = [self.fuzzy_search(query, limit=10) for query in queries]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Cache warming completed")