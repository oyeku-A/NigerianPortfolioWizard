import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and cleans Nigerian market data."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.max_price_change = 0.20  # Maximum allowed daily price change (20%)
        
    def validate_returns(self, returns_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate returns data with enhanced outlier detection for Nigerian markets.
        
        Args:
            returns_df (pd.DataFrame): Returns data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing values
        missing_values = returns_df.isnull().sum()
        if missing_values.any():
            issues.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
        
        # Enhanced outlier detection for Nigerian markets
        # Nigerian stocks can have higher volatility, but we still need reasonable limits
        extreme_threshold = 0.20  # Reduced from 0.50 to 20% daily return threshold
        extreme_returns = returns_df[abs(returns_df) > extreme_threshold]
        if not extreme_returns.empty:
            issues.append(f"Extreme returns detected (>20%): {len(extreme_returns)} instances")
        
        # Check for consistent extreme returns (potential data issues)
        for col in returns_df.columns:
            col_returns = returns_df[col]
            extreme_count = len(col_returns[abs(col_returns) > extreme_threshold])
            if extreme_count > len(col_returns) * 0.05:  # More than 5% extreme returns (reduced from 10%)
                issues.append(f"Column {col} has {extreme_count} extreme returns ({extreme_count/len(col_returns)*100:.1f}%)")
        
        # Check for zero variance (stocks with no price movement)
        zero_variance_cols = returns_df.columns[returns_df.var() == 0]
        if len(zero_variance_cols) > 0:
            issues.append(f"Zero variance detected in columns: {list(zero_variance_cols)}")
        
        # Check for infinite values
        if np.isinf(returns_df.values).any():
            issues.append("Infinite values detected in returns data")
        
        # Check for annualized returns that are too high
        daily_means = returns_df.mean()
        annualized_returns = daily_means * 252  # Assuming 252 trading days
        unrealistic_returns = annualized_returns[annualized_returns > 0.50]  # 50% annual return limit
        if not unrealistic_returns.empty:
            issues.append(f"Unrealistic annualized returns detected: {unrealistic_returns.to_dict()}")
        
        return len(issues) == 0, issues
        
    def validate_covariance(self, cov_matrix: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate covariance matrix.
        
        Args:
            cov_matrix (pd.DataFrame): Covariance matrix to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing values
        if cov_matrix.isnull().any().any():
            issues.append("Missing values found in covariance matrix")
            
        # Check for symmetry
        if not np.allclose(cov_matrix, cov_matrix.T):
            issues.append("Covariance matrix is not symmetric")
            
        # Check for positive definiteness
        try:
            np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            issues.append("Covariance matrix is not positive definite")
            
        return len(issues) == 0, issues 