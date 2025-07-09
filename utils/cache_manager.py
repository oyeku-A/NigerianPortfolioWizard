import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages local caching of market data and optimization results."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize the CacheManager.
        
        Args:
            cache_dir (str): Directory for caching data
        """
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get full path for cache file."""
        return os.path.join(self.cache_dir, f"{key}.csv")
        
    def _get_metadata_path(self, key: str) -> str:
        """Get full path for metadata file."""
        return os.path.join(self.cache_dir, f"{key}_metadata.json")
        
    def is_cached(self, key: str, max_age_days: int = 1) -> bool:
        """
        Check if data is cached and not expired.
        
        Args:
            key (str): Cache key
            max_age_days (int): Maximum age of cache in days
            
        Returns:
            bool: True if valid cache exists
        """
        metadata_path = self._get_metadata_path(key)
        if not os.path.exists(metadata_path):
            return False
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            cache_time = datetime.fromisoformat(metadata['timestamp'])
            age = datetime.now() - cache_time
            
            return age.days < max_age_days
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
            
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[pd.DataFrame]: Cached data if exists
        """
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None
            
        try:
            # Read CSV with explicit date parsing
            df = pd.read_csv(cache_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
            
    def cache_data(self, key: str, data: pd.DataFrame, metadata: Dict[str, Any] = None):
        """
        Cache data with metadata.
        
        Args:
            key (str): Cache key
            data (pd.DataFrame): Data to cache
            metadata (Dict[str, Any], optional): Additional metadata
        """
        try:
            # Save data
            cache_path = self._get_cache_path(key)
            # Reset index to ensure date is saved as a column
            data.reset_index().to_csv(cache_path, index=False)
            
            # Save metadata
            metadata = metadata or {}
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['rows'] = len(data)
            metadata['columns'] = list(data.columns)
            
            metadata_path = self._get_metadata_path(key)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Cached data for key: {key}")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            
    def clear_cache(self, key: Optional[str] = None):
        """
        Clear cache for specific key or all cache.
        
        Args:
            key (Optional[str]): Cache key to clear, or None for all cache
        """
        try:
            if key:
                # Clear specific cache
                cache_path = self._get_cache_path(key)
                metadata_path = self._get_metadata_path(key)
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    
                logger.info(f"Cleared cache for key: {key}")
            else:
                # Clear all cache
                for file in os.listdir(self.cache_dir):
                    os.remove(os.path.join(self.cache_dir, file))
                logger.info("Cleared all cache")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dict[str, Any]: Cache information
        """
        cache_info = {}
        
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('_metadata.json'):
                    key = file.replace('_metadata.json', '')
                    metadata_path = os.path.join(self.cache_dir, file)
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    cache_info[key] = {
                        'timestamp': metadata['timestamp'],
                        'rows': metadata['rows'],
                        'columns': metadata['columns']
                    }
                    
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            
        return cache_info 