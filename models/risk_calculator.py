"""
Risk calculation module for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nigerian_constants import RISK_METRICS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskCalculator:
    """Calculates Nigeria-specific risk metrics for portfolios."""
    
    def __init__(self, prices_df: pd.DataFrame, returns_df: pd.DataFrame):
        """
        Initialize the RiskCalculator.
        
        Args:
            prices_df (pd.DataFrame): Historical price data
            returns_df (pd.DataFrame): Historical returns data
        """
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.symbols = returns_df.columns.tolist()
        
    def calculate_var(self, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            weights: Array of portfolio weights
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Annualized VaR value
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns_df.dot(weights)
            
            # Calculate daily VaR
            daily_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # Annualize VaR (assuming normal distribution)
            annualized_var = abs(daily_var) * np.sqrt(252)
            
            # Cap at reasonable levels (not exceeding 100%)
            annualized_var = min(annualized_var, 1.0)
            
            return annualized_var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
        
    def calculate_cvar(self, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) for the portfolio.
        
        Args:
            weights: Array of portfolio weights
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            Annualized CVaR value
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns_df.dot(weights)
            
            # Calculate VaR threshold
            var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # Calculate daily CVaR (average of returns below VaR threshold)
            daily_cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
            
            # Annualize CVaR (assuming normal distribution)
            annualized_cvar = abs(daily_cvar) * np.sqrt(252)
            
            # Cap at reasonable levels (not exceeding 100%)
            annualized_cvar = min(annualized_cvar, 1.0)
            
            return annualized_cvar
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
        
    def calculate_max_drawdown(self, weights: np.ndarray) -> float:
        """
        Calculate maximum drawdown for the portfolio.
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            Maximum drawdown value
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns_df.dot(weights)
            
            # Calculate cumulative returns
            cum_returns = (1 + portfolio_returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cum_returns - running_max) / running_max
            
            # Get maximum drawdown
            max_drawdown = abs(drawdowns.min())
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return 0.0
        
    def calculate_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            Annualized volatility (always computed from returns)
        """
        try:
            portfolio_returns = self.returns_df.dot(weights)
            volatility = portfolio_returns.std() * np.sqrt(252)
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return np.nan
        
    def calculate_beta(self, weights: np.ndarray, market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to market.
        
        Args:
            weights: Array of portfolio weights
            market_returns: Series of market returns
            
        Returns:
            Portfolio beta
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns_df.dot(weights)
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance
            
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0
        
    def calculate_correlation(self, weights: np.ndarray, market_returns: pd.Series) -> float:
        """
        Calculate portfolio correlation with market.
        
        Args:
            weights: Array of portfolio weights
            market_returns: Series of market returns
            
        Returns:
            Portfolio correlation
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns_df.dot(weights)
            
            # Calculate correlation
            correlation = np.corrcoef(portfolio_returns, market_returns)[0, 1]
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
        
    def calculate_sector_concentration(self, weights: np.ndarray, sector_info: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate sector concentration for the portfolio.
        
        Args:
            weights: Array of portfolio weights
            sector_info: Dictionary mapping symbols to sectors
            
        Returns:
            Dictionary of sector weights (as percentages)
        """
        try:
            # Create sector weights dictionary
            sector_weights = {}
            
            # Ensure we have the correct symbols that match the weights
            symbols = self.symbols[:len(weights)] if len(self.symbols) >= len(weights) else self.symbols
            
            # Calculate weights for each sector
            for i, symbol in enumerate(symbols):
                if i < len(weights):
                    sector = sector_info.get(symbol, 'Other')
                    sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
            
            # Ensure weights sum to 1 (100%)
            total_weight = sum(sector_weights.values())
            if total_weight > 0:
                sector_weights = {sector: weight / total_weight for sector, weight in sector_weights.items()}
            
            return sector_weights
            
        except Exception as e:
            logger.error(f"Error calculating sector concentration: {str(e)}")
            return {}
        
    def calculate_liquidity_risk(self, weights: np.ndarray, volume_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate liquidity risk for the portfolio using average daily trading volume.
        Lower average volume means higher liquidity risk.
        Args:
            weights: Array of portfolio weights
            volume_data: DataFrame of daily trading volumes (same columns as prices_df)
        Returns:
            Liquidity risk score (0 = most liquid, 1 = least liquid) or np.nan if unavailable
        """
        try:
            if volume_data is None or volume_data.empty:
                logger.warning("Liquidity risk cannot be computed: volume data missing.")
                return np.nan  # Cannot compute without volume data
            avg_volumes = volume_data.mean()
            median_volume = np.median(avg_volumes)
            weighted_avg_volume = np.sum(weights * avg_volumes.loc[self.symbols[:len(weights)]].values)
            liquidity_score = 1.0 - (weighted_avg_volume / (median_volume + 1e-9))
            liquidity_score = float(np.clip(liquidity_score, 0, 1))
            return liquidity_score
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {str(e)}")
            return np.nan
        
    def get_risk_metrics(self, weights: np.ndarray, 
                        sector_info: Optional[Dict[str, str]] = None,
                        volume_data: Optional[pd.DataFrame] = None,
                        currency_exposure: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for the portfolio.
        
        Args:
            weights: Array of portfolio weights
            sector_info (Optional[Dict[str, str]]): Sector information
            volume_data (Optional[pd.DataFrame]): Volume data
            currency_exposure (Optional[Dict[str, float]]): Currency exposure
            
        Returns:
            Dict[str, float]: Comprehensive risk metrics
        """
        try:
            metrics = {
                'var_95': self.calculate_var(weights, RISK_METRICS['var_confidence_level']),
                'cvar_95': self.calculate_cvar(weights, RISK_METRICS['cvar_confidence_level']),
                'max_drawdown': self.calculate_max_drawdown(weights),
                'volatility': self.calculate_volatility(weights)
            }
            
            if hasattr(self, 'market_returns'):
                metrics.update({
                    'beta': self.calculate_beta(weights, self.market_returns),
                    'correlation': self.calculate_correlation(weights, self.market_returns)
                })
            
            if sector_info:
                metrics['sector_concentration'] = self.calculate_sector_concentration(weights, sector_info)
            
            if volume_data is not None and not volume_data.empty:
                metrics['liquidity_risk'] = self.calculate_liquidity_risk(weights, volume_data)
            else:
                metrics['liquidity_risk'] = 'N/A'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'beta': 1.0,
                'correlation': 0.0,
                'sector_concentration': {},
                'liquidity_risk': 'N/A'
            }
    
    def set_market_returns(self, market_returns: pd.Series) -> None:
        """
        Set market returns for beta and correlation calculations.
        
        Args:
            market_returns: Series of market returns
        """
        self.market_returns = market_returns 