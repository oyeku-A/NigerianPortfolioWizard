import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional, Tuple
import logging
import json
from dotenv import load_dotenv
from utils.cache_manager import CacheManager
from utils.data_validator import DataValidator
from utils.nigerian_constants import MIN_PRICE_HISTORY_DAYS
from utils.nigerian_sectors import NIGERIAN_STOCK_SECTORS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data acquisition and preprocessing for Nigerian market data."""
    
    def __init__(self, api_key: str, cache_dir: str = "data/raw"):
        """
        Initialize the DataLoader.
        
        Args:
            api_key (str): API key for data provider
            cache_dir (str): Directory for caching data
        """
        self.api_key = api_key
        self.cache_manager = CacheManager(cache_dir)
        self.data_validator = DataValidator()
        
    def get_nigerian_stocks(self, use_cache: bool = True, max_age_days: int = 1) -> pd.DataFrame:
        """
        Fetch list of Nigerian stocks from EODHD. Uses caching.
        
        Args:
            use_cache (bool): Whether to use cached data if available and not expired.
            max_age_days (int): Maximum age of cached data in days.
            
        Returns:
            pd.DataFrame: DataFrame containing stock information (Code, Name, Currency, Type, Sector).
        """
        cache_key = "nigerian_stocks_list"
        
        if use_cache and self.cache_manager.is_cached(cache_key, max_age_days):
            logger.info("Loading Nigerian stocks from cache.")
            cached_data = self.cache_manager.get_cached_data(cache_key)
            if cached_data is not None and not cached_data.empty:
                return cached_data
            logger.warning("Cached stock list is empty or corrupted, refetching.")
            
        logger.info("Fetching Nigerian stocks from EODHD API.")
        url = "https://eodhd.com/api/exchange-symbol-list/XNSA"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "type": "common_stock",
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning("EODHD API returned no stock data for XNSA.")
                return pd.DataFrame()

            # Extract unique stocks
            stocks_data = []
            for s in data:
                stock_info = {
                    'Code': s.get('Code'),
                    'Name': s.get('Name'),
                    'Currency': s.get('Currency'),
                    'Type': s.get('Type'),
                    'Exchange': s.get('Exchange')
                }
                stocks_data.append(stock_info)
            
            stocks_df = pd.DataFrame(stocks_data)
            stocks_df = stocks_df.drop_duplicates(subset=['Code'])

            # Map sectors using pre-defined mapping from nigerian_sectors.py
            logger.info("Mapping sectors using pre-defined mapping...")
            stocks_df['Sector'] = stocks_df['Code'].map(NIGERIAN_STOCK_SECTORS).fillna('Other')

            # Basic validation for essential columns
            required_cols = ['Code', 'Name', 'Currency', 'Type', 'Sector']
            if not all(col in stocks_df.columns for col in required_cols):
                logger.error(f"Missing essential columns in fetched stock data: {required_cols}")
                return pd.DataFrame()

            self.cache_manager.cache_data(cache_key, stocks_df)
            logger.info(f"Successfully fetched and cached {len(stocks_df)} Nigerian stocks with sector information.")
            return stocks_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or API error fetching Nigerian stocks: {e}")
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON response for Nigerian stocks.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching Nigerian stocks: {e}")
            
        return pd.DataFrame()
            
    def fetch_stock_data(self, symbols: List[str], days: int = MIN_PRICE_HISTORY_DAYS, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical price and volume data for specified symbols from EODHD. Uses caching and data validation.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (prices_df, volume_df)
        """
        if not symbols:
            return pd.DataFrame(), pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_prices_df = pd.DataFrame()
        all_volumes_df = pd.DataFrame()
        fetched_new_data = False

        for symbol in symbols:
            cache_key = f"eod_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            if use_cache and self.cache_manager.is_cached(cache_key, max_age_days=1):
                logger.info(f"Loading historical data for {symbol} from cache.")
                stock_df = self.cache_manager.get_cached_data(cache_key)
                if stock_df is not None and not stock_df.empty:
                    if 'close' in stock_df.columns and 'volume' in stock_df.columns:
                        all_prices_df = pd.concat([all_prices_df, stock_df[['close']].rename(columns={'close': symbol})], axis=1)
                        all_volumes_df = pd.concat([all_volumes_df, stock_df[['volume']].rename(columns={'volume': symbol})], axis=1)
                    continue
                logger.warning(f"Cached data for {symbol} is empty or corrupted, refetching.")

            logger.info(f"Fetching historical data for {symbol} from EODHD API.")
            url = f"https://eodhd.com/api/eod/{symbol}.XNSA"
            params = {
                "api_token": self.api_key,
                "fmt": "json",
                "from": start_date.strftime('%Y-%m-%d'),
                "to": end_date.strftime('%Y-%m-%d')
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data or not isinstance(data, list):
                    logger.warning(f"No valid historical data for {symbol}.")
                    continue
                    
                # Convert the data to DataFrame
                stock_df = pd.DataFrame(data)
                
                # Check if required columns exist
                if 'date' not in stock_df.columns or 'close' not in stock_df.columns or 'volume' not in stock_df.columns:
                    logger.error(f"Missing required columns for {symbol}. Available columns: {stock_df.columns.tolist()}")
                    continue
                
                # Process the data
                stock_df['date'] = pd.to_datetime(stock_df['date'])
                stock_df.set_index('date', inplace=True)
                
                # Keep only the close price and volume, rename columns to the symbol
                all_prices_df = pd.concat([all_prices_df, stock_df[['close']].rename(columns={'close': symbol})], axis=1)
                all_volumes_df = pd.concat([all_volumes_df, stock_df[['volume']].rename(columns={'volume': symbol})], axis=1)
                
                # Cache the processed data
                self.cache_manager.cache_data(cache_key, stock_df)
                fetched_new_data = True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network or API error fetching data for {symbol}: {e}")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response for {symbol}.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while fetching data for {symbol}: {e}")
                if 'response' in locals():
                    logger.error(f"Response content: {response.text}")
        
        if all_prices_df.empty:
            logger.error("No historical data could be fetched for any of the selected symbols.")
            return pd.DataFrame(), pd.DataFrame()

        # Ensure all columns are numeric, coerce errors
        for col in all_prices_df.columns:
            all_prices_df[col] = pd.to_numeric(all_prices_df[col], errors='coerce')
        for col in all_volumes_df.columns:
            all_volumes_df[col] = pd.to_numeric(all_volumes_df[col], errors='coerce')
        
        # Drop any symbols for which fetching completely failed or data is all NaN
        all_prices_df.dropna(axis=1, how='all', inplace=True)
        all_volumes_df.dropna(axis=1, how='all', inplace=True)
        
        return all_prices_df, all_volumes_df
        
    def calculate_returns_and_cov(self, prices_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate returns and covariance matrix from price data.
        
        Args:
            prices_df (pd.DataFrame): DataFrame of historical prices.
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: (annual_returns, covariance_matrix).
        """
        logger.info("Calculating returns and covariance matrix.")
        
        if prices_df.empty:
            logger.warning("Price data is empty, cannot calculate returns and covariance.")
            return pd.Series(), pd.DataFrame()

        # Ensure prices are sorted by date and handle duplicates
        prices_df = prices_df.sort_index().drop_duplicates()

        # More aggressive data cleaning for Nigerian market data
        # Remove columns with too many NaN values (>50%)
        nan_threshold = 0.5
        valid_columns = []
        for col in prices_df.columns:
            nan_ratio = prices_df[col].isna().sum() / len(prices_df)
            if nan_ratio <= nan_threshold:
                valid_columns.append(col)
            else:
                logger.warning(f"Dropping {col} due to {nan_ratio:.1%} missing values")
        
        if len(valid_columns) < 2:
            logger.error("Insufficient valid columns for portfolio optimization")
            return pd.Series(), pd.DataFrame()
        
        prices_df = prices_df[valid_columns]

        # Fill missing values using forward fill then backward fill
        prices_df = prices_df.ffill().bfill()
        
        # If any NaN values remain, fill with column mean
        prices_df = prices_df.fillna(prices_df.mean())
        
        # Additional validation: check for zero or negative prices
        for col in prices_df.columns:
            if (prices_df[col] <= 0).any():
                logger.warning(f"Found zero or negative prices in {col}, replacing with previous valid price")
                prices_df[col] = prices_df[col].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
                prices_df[col] = prices_df[col].ffill().bfill()
                # If still NaN, use the mean
                if prices_df[col].isna().any():
                    prices_df[col] = prices_df[col].fillna(prices_df[col].mean())
        
        # Remove stocks with zero variance (no price movement)
        zero_variance_cols = []
        for col in prices_df.columns:
            if prices_df[col].var() == 0 or prices_df[col].std() < 1e-6:
                zero_variance_cols.append(col)
                logger.warning(f"Removing {col} due to zero variance (no price movement)")
        
        if zero_variance_cols:
            prices_df = prices_df.drop(columns=zero_variance_cols)
            if len(prices_df.columns) < 2:
                logger.error("Insufficient stocks with price movement for portfolio optimization")
                return pd.Series(), pd.DataFrame()
        
        # Calculate returns with better handling of edge cases
        try:
            returns_df = prices_df.pct_change(fill_method=None).dropna()
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(), pd.DataFrame()

        # More aggressive cleaning of extreme returns
        if not returns_df.empty:
            # Clip extreme returns to ±20% (more conservative for Nigerian markets)
            returns_df = returns_df.clip(lower=-0.20, upper=0.20)
            
            # Remove any remaining NaN values
            returns_df = returns_df.dropna()
            
            # Check for infinite values
            if np.isinf(returns_df.values).any():
                logger.warning("Found infinite values in returns, replacing with NaN")
                returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna()

        if returns_df.empty:
            logger.error("Returns DataFrame is empty after calculation and cleaning.")
            return pd.Series(), pd.DataFrame()

        # Validate returns data
        is_valid_returns, returns_issues = self.data_validator.validate_returns(returns_df)
        if not is_valid_returns:
            logger.warning(f"Returns data validation issues: {returns_issues}")

        # Annualize returns (using constant from nigerian_constants)
        daily_returns_mean = returns_df.mean()
        annual_returns = daily_returns_mean * MIN_PRICE_HISTORY_DAYS
        
        # More aggressive capping of unrealistic returns
        max_reasonable_return = 0.50  # 50% annual return (more conservative)
        if (annual_returns > max_reasonable_return).any():
            logger.warning(f"Unrealistic annual returns detected: {annual_returns[annual_returns > max_reasonable_return]}")
            # Cap returns at reasonable levels
            annual_returns = annual_returns.clip(upper=max_reasonable_return)
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_df.cov() * MIN_PRICE_HISTORY_DAYS

        # Validate covariance matrix and apply regularization if needed
        is_valid_cov, cov_issues = self.data_validator.validate_covariance(cov_matrix)
        if not is_valid_cov:
            logger.warning(f"Covariance matrix validation issues: {cov_issues}. Applying regularization.")
            # Apply more aggressive regularization for Nigerian market data
            regularization_factor = 1e-3  # Increased from 1e-4
            cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * regularization_factor
            
            # Verify positive definiteness after regularization
            try:
                np.linalg.cholesky(cov_matrix)
                logger.info("Covariance matrix is now positive definite after regularization")
            except np.linalg.LinAlgError:
                logger.error("Covariance matrix still not positive definite after regularization")
                # Use diagonal matrix as last resort
                cov_matrix = np.diag(np.diag(cov_matrix)) + np.eye(len(cov_matrix)) * 1e-2

        logger.info("Returns and covariance matrix calculated successfully.")
        return annual_returns, cov_matrix
        
    def get_market_data(self, symbols: List[str], days: int = MIN_PRICE_HISTORY_DAYS) -> Optional[Dict]:
        """
        Get complete market data for specified symbols.
        Returns:
            Optional[Dict]: Dictionary containing prices, returns, covariance matrix, daily_returns, and volume_data (if available).
        """
        if not symbols:
            logger.warning("No symbols provided for get_market_data.")
            return None

        if len(symbols) < 2:
            logger.warning("At least 2 symbols required for portfolio optimization.")
            return None

        prices_df, volume_df = self.fetch_stock_data(symbols, days=days)
        if prices_df.empty:
            logger.error("Failed to fetch any price data.")
            return None
        if len(prices_df.columns) < 2:
            logger.error(f"Failed to fetch data for multiple symbols. Got {len(prices_df.columns)} symbols, need at least 2.")
            return None

        # Check data quality before proceeding
        data_quality_issues = []
        
        # Check for sufficient data points
        if len(prices_df) < 30:  # At least 30 days of data
            data_quality_issues.append(f"Insufficient data points: {len(prices_df)} days (minimum 30 required)")
        
        # Check for stocks with no price movement
        zero_variance_stocks = []
        for col in prices_df.columns:
            if prices_df[col].var() == 0 or prices_df[col].std() < 1e-6:
                zero_variance_stocks.append(col)
        
        if zero_variance_stocks:
            data_quality_issues.append(f"Stocks with no price movement: {zero_variance_stocks}")
            # Remove these stocks
            prices_df = prices_df.drop(columns=zero_variance_stocks)
            if len(prices_df.columns) < 2:
                logger.error("Insufficient stocks with price movement for portfolio optimization.")
                return None
        
        if data_quality_issues:
            logger.warning(f"Data quality issues detected: {data_quality_issues}")

        # Calculate returns and covariance
        annual_returns, cov_matrix = self.calculate_returns_and_cov(prices_df)
        
        if annual_returns.empty or cov_matrix.empty:
            logger.error("Failed to calculate returns and covariance matrix.")
            return None

        # Final validation of calculated metrics
        if (annual_returns > 0.50).any():
            logger.warning("Some stocks have unrealistic annual returns (>50%), capping at 50%")
            annual_returns = annual_returns.clip(upper=0.50)
        
        # Calculate daily returns for risk calculations
        prices_clean = prices_df.sort_index().drop_duplicates().ffill().bfill().fillna(prices_df.mean())
        daily_returns = prices_clean.pct_change(fill_method=None).dropna()
        
        # Clean daily returns as well
        daily_returns = daily_returns.clip(lower=-0.20, upper=0.20)

        logger.info(f"Successfully processed market data for {len(prices_df.columns)} symbols with {len(prices_df)} days of data")
        
        return {
            'prices': prices_df,
            'returns': annual_returns,
            'daily_returns': daily_returns,
            'covariance': cov_matrix,
            'volume_data': volume_df
        }
        
    def augment_limited_data(self, prices_df: pd.DataFrame, target_days: int = 500) -> pd.DataFrame:
        """
        Augment limited data using bootstrapping and volatility scaling.
        Useful when only 1 year of data is available.
        Args:
            prices_df (pd.DataFrame): Original price data
            target_days (int): Target number of days for augmentation
        Returns:
            pd.DataFrame: Augmented price data
        """
        if prices_df.empty:
            return prices_df
        logger.info(f"Augmenting {len(prices_df)} days of data to {target_days} days")
        returns_df = prices_df.pct_change(fill_method=None).dropna()
        if returns_df.empty:
            return prices_df
        zero_variance_cols = [col for col in returns_df.columns if returns_df[col].var() == 0 or returns_df[col].std() < 1e-6]
        if zero_variance_cols:
            returns_df = returns_df.drop(columns=zero_variance_cols)
            if len(returns_df.columns) < 2:
                logger.warning("Insufficient stocks for augmentation, returning original data")
                return prices_df
        # Clip extreme returns before augmentation
        returns_df = returns_df.clip(lower=-0.10, upper=0.10)
        augmented_returns = []
        current_length = len(returns_df)
        chunk_size = min(30, current_length)
        while len(augmented_returns) < target_days:
            remaining_days = target_days - len(augmented_returns)
            sample_size = min(chunk_size, remaining_days)
            sample_indices = np.random.choice(current_length, size=sample_size, replace=True)
            chunk_returns = returns_df.iloc[sample_indices].values.tolist()
            for i in range(len(chunk_returns)):
                for j in range(len(chunk_returns[i])):
                    chunk_returns[i][j] = np.clip(chunk_returns[i][j], -0.10, 0.10)
            augmented_returns.extend(chunk_returns)
        augmented_returns = augmented_returns[:target_days]
        augmented_returns_df = pd.DataFrame(augmented_returns, columns=returns_df.columns, index=range(len(augmented_returns)))
        initial_prices = prices_df.iloc[0]
        augmented_prices = [initial_prices]
        for returns_row in augmented_returns_df.values:
            new_prices = augmented_prices[-1] * (1 + returns_row)
            new_prices = np.maximum(new_prices, 0.01)
            new_prices = np.minimum(new_prices, initial_prices * 10)
            augmented_prices.append(new_prices)
        augmented_prices_df = pd.DataFrame(augmented_prices, columns=returns_df.columns)
        # Post-augmentation validation
        final_returns = augmented_prices_df.pct_change(fill_method=None).dropna()
        min_ret, max_ret, mean_ret = None, None, None
        if not final_returns.empty:
            min_ret = final_returns.min().min()
            max_ret = final_returns.max().max()
            mean_ret = final_returns.mean().mean()
            extreme_count = (abs(final_returns) > 0.10).sum().sum()
            if extreme_count > 0:
                logger.warning(f"Augmentation created {extreme_count} extreme returns, applying final cleanup")
                final_returns = final_returns.clip(lower=-0.10, upper=0.10)
                cleaned_prices = [initial_prices]
                for returns_row in final_returns.values:
                    new_prices = cleaned_prices[-1] * (1 + returns_row)
                    new_prices = np.maximum(new_prices, 0.01)
                    new_prices = np.minimum(new_prices, initial_prices * 10)
                    cleaned_prices.append(new_prices)
                augmented_prices_df = pd.DataFrame(cleaned_prices, columns=returns_df.columns)
                # Recompute stats
                final_returns = augmented_prices_df.pct_change(fill_method=None).dropna()
                min_ret = final_returns.min().min()
                max_ret = final_returns.max().max()
                mean_ret = final_returns.mean().mean()
        logger.info(f"Augmented data stats: min={min_ret:.4f}, max={max_ret:.4f}, mean={mean_ret:.4f}")
        # Attach stats for UI display (as attribute)
        augmented_prices_df.aug_stats = {'min': min_ret, 'max': max_ret, 'mean': mean_ret}
        return augmented_prices_df 