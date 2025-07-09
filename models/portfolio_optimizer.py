"""
Portfolio optimization module with improved numerical stability and convergence.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, returns: pd.Series, covariance: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns: Series of asset returns
            covariance: DataFrame of asset covariances
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.returns = returns
        self.cov_matrix = covariance
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns)
        
        # Add small regularization to covariance matrix for numerical stability
        self.cov_matrix = self.cov_matrix + np.eye(self.num_assets) * 1e-6
        
    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio statistics with proper annualization (assumes annualized returns/covariance)."""
        portfolio_return = np.sum(self.returns * weights)  # already annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))  # already annualized
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective function for Sharpe ratio maximization."""
        return -self._portfolio_stats(weights)[2]
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Objective function for volatility minimization."""
        return self._portfolio_stats(weights)[1]
    
    def _constraints(self, target_return: Optional[float] = None, 
                    constraints: Optional[Dict] = None,
                    sector_info: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Generate optimization constraints."""
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.returns * x) - target_return  # Use daily returns for constraint
            })
        
        # Add sector concentration constraints if sector info is provided
        if sector_info:
            # Group assets by sector
            sector_groups = {}
            for i, symbol in enumerate(self.returns.index):
                sector = sector_info.get(symbol, 'Other')
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(i)
            
            # Add constraint for each sector (max 50% per sector)
            for sector, indices in sector_groups.items():
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x, idxs=indices: 0.5 - np.sum(x[idxs])  # max 50% per sector
                })
        
        return constraints_list
    
    def _bounds(self, constraints: Optional[Dict] = None) -> List[Tuple[float, float]]:
        """Generate bounds for weights."""
        if constraints and 'max_stock_weight' in constraints:
            max_weight = constraints['max_stock_weight']
        else:
            max_weight = 0.25  # Reduced from 0.4 to 25% max per asset for better diversification
            
        return [(0, max_weight) for _ in range(self.num_assets)]
    
    def optimize(self, risk_level: str = 'medium', 
                constraints: Optional[Dict] = None,
                sector_info: Optional[Dict[str, str]] = None) -> Dict:
        """
        Optimize portfolio weights based on risk level.
        
        Args:
            risk_level: 'low', 'medium', or 'high'
            constraints: Dictionary of optimization constraints
            sector_info: Dictionary mapping symbols to sectors for sector constraints
        
        Returns:
            Dictionary containing optimization results
        """
        # Define risk levels and their target returns (annualized)
        risk_levels = {
            'low': 0.08,     # 8% annual target return
            'medium': 0.12,  # 12% annual target return
            'high': 0.16     # 16% annual target return
        }
        
        target_return_annual = risk_levels.get(risk_level, 0.12)
        
        # Initial guess - equal weights
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimization options
        options = {
            'maxiter': 1000,
            'ftol': 1e-8,
            'disp': False
        }
        
        try:
            # Try maximizing Sharpe ratio first
            result = minimize(
                self._negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=self._bounds(constraints),
                constraints=self._constraints(target_return_annual, constraints, sector_info),
                options=options
            )
            
            if not result.success:
                # If Sharpe maximization fails, try minimizing volatility
                result = minimize(
                    self._portfolio_volatility,
                    initial_weights,
                    method='SLSQP',
                    bounds=self._bounds(constraints),
                    constraints=self._constraints(target_return_annual, constraints, sector_info),
                    options=options
                )
            
            if not result.success:
                logger.warning(f"Optimization for {risk_level} risk level did not converge: {result.message}")
                return {
                    'weights': initial_weights,
                    'expected_return': self._portfolio_stats(initial_weights)[0],
                    'volatility': self._portfolio_stats(initial_weights)[1],
                    'sharpe_ratio': self._portfolio_stats(initial_weights)[2],
                    'method': 'equal_weight_fallback'
                }
            
            return {
                'weights': result.x,
                'expected_return': self._portfolio_stats(result.x)[0],
                'volatility': self._portfolio_stats(result.x)[1],
                'sharpe_ratio': self._portfolio_stats(result.x)[2],
                'method': 'optimized'
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'weights': initial_weights,
                'expected_return': self._portfolio_stats(initial_weights)[0],
                'volatility': self._portfolio_stats(initial_weights)[1],
                'sharpe_ratio': self._portfolio_stats(initial_weights)[2],
                'method': 'equal_weight_fallback'
            }
    
    def generate_efficient_frontier(self, num_portfolios: int = 100, sector_info: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios using Monte Carlo simulation with sector constraints.
        Optimized for limited data scenarios (1 year of data).
        
        Args:
            num_portfolios: Number of portfolios to generate
            sector_info: Dictionary mapping symbols to sectors for constraints
        
        Returns:
            DataFrame containing efficient frontier portfolios
        """
        frontier_portfolios = []
        
        # For limited data scenarios, use more conservative constraints
        max_weight = 0.30  # Reduced from 0.35 to 30% for better diversification
        max_sector_weight = 0.50  # Reduced from 0.60 to 50% for better sector diversification
        
        # Generate random portfolios using Monte Carlo simulation
        attempts = 0
        max_attempts = num_portfolios * 3  # Allow more attempts to find valid portfolios
        
        while len(frontier_portfolios) < num_portfolios and attempts < max_attempts:
            attempts += 1
            
            try:
                # Generate random weights that sum to 1
                weights = np.random.random(self.num_assets)
                weights = weights / np.sum(weights)  # Normalize to sum to 1
                
                # Apply individual asset constraints
                weights = np.clip(weights, 0, max_weight)
                weights = weights / np.sum(weights)  # Renormalize
                
                # Apply sector constraints if available
                if sector_info and len(sector_info) > 0:
                    sector_groups = {}
                    for i, symbol in enumerate(self.returns.index):
                        sector = sector_info.get(symbol, 'Other')
                        if sector not in sector_groups:
                            sector_groups[sector] = []
                        sector_groups[sector].append(i)
                    
                    # Check if any sector exceeds the maximum weight
                    valid_weights = True
                    for sector, indices in sector_groups.items():
                        sector_weight = np.sum(weights[indices])
                        if sector_weight > max_sector_weight:
                            valid_weights = False
                            break
                    
                    if not valid_weights:
                        continue  # Skip this portfolio
                
                # Calculate portfolio statistics
                portfolio_return, portfolio_vol, sharpe_ratio = self._portfolio_stats(weights)
                
                # More realistic bounds for Nigerian market data
                # Annual volatility: 15% to 80% (more conservative)
                # Annual returns: -40% to +80% (more realistic for Nigerian market)
                if (portfolio_vol > 0.15 and portfolio_vol < 0.80 and 
                    portfolio_return > -0.40 and portfolio_return < 0.80 and
                    not np.isnan(portfolio_return) and not np.isnan(portfolio_vol)):
                    
                    frontier_portfolios.append({
                        'Return': portfolio_return,
                        'Volatility': portfolio_vol,
                        'Sharpe': sharpe_ratio,
                        'Weights': weights.copy()  # Make a copy to avoid reference issues
                    })
                
            except Exception as e:
                logger.debug(f"Failed to calculate portfolio stats: {e}")
                continue
        
        if not frontier_portfolios:
            logger.warning("No portfolios generated with constraints, trying without sector constraints...")
            # Fallback: generate portfolios without sector constraints but with individual asset limits
            for _ in range(num_portfolios // 2):
                try:
                    weights = np.random.random(self.num_assets)
                    weights = weights / np.sum(weights)
                    weights = np.clip(weights, 0, 0.35)  # 35% max per asset
                    weights = weights / np.sum(weights)
                    
                    portfolio_return, portfolio_vol, sharpe_ratio = self._portfolio_stats(weights)
                    
                    if (portfolio_vol > 0.15 and portfolio_vol < 1.0 and 
                        not np.isnan(portfolio_return) and not np.isnan(portfolio_vol)):
                        frontier_portfolios.append({
                            'Return': portfolio_return,
                            'Volatility': portfolio_vol,
                            'Sharpe': sharpe_ratio,
                            'Weights': weights.copy()
                        })
                        
                except Exception as e:
                    continue
        
        if not frontier_portfolios:
            logger.error("No efficient frontier portfolios could be generated even with relaxed constraints")
            # Generate simple equal-weight portfolio as last resort
            try:
                weights = np.array([1/self.num_assets] * self.num_assets)
                portfolio_return, portfolio_vol, sharpe_ratio = self._portfolio_stats(weights)
                
                frontier_portfolios.append({
                    'Return': portfolio_return,
                    'Volatility': portfolio_vol,
                    'Sharpe': sharpe_ratio,
                    'Weights': weights
                })
                logger.info("Generated equal-weight portfolio as fallback")
            except Exception as e:
                logger.error(f"Failed to generate even equal-weight portfolio: {e}")
                return pd.DataFrame()
        
        frontier_df = pd.DataFrame(frontier_portfolios)
        
        # Sort by volatility and remove duplicates
        frontier_df = frontier_df.sort_values('Volatility').drop_duplicates(subset=['Volatility'])
        
        # Keep only the portfolios with the highest return for each volatility level (efficient frontier)
        efficient_portfolios = []
        if len(frontier_df) > 5:  # Only bin if we have enough portfolios
            try:
                volatility_bins = pd.cut(frontier_df['Volatility'], bins=min(15, len(frontier_df)//3))
                for _, group in frontier_df.groupby(volatility_bins, observed=True):
                    if not group.empty:
                        # Find the portfolio with highest return in this volatility range
                        best_portfolio = group.loc[group['Return'].idxmax()]
                        efficient_portfolios.append(best_portfolio)
            except Exception as e:
                logger.warning(f"Error in binning portfolios: {e}, using all portfolios")
                efficient_portfolios = frontier_df.to_dict('records')
        else:
            # If too few portfolios, just return the sorted frontier
            efficient_portfolios = frontier_df.to_dict('records')
        
        if efficient_portfolios:
            efficient_df = pd.DataFrame(efficient_portfolios)
            return efficient_df.sort_values('Volatility')
        else:
            return frontier_df.sort_values('Volatility')
    
    def generate_simple_portfolio(self) -> Dict:
        """
        Generate a simple equal-weight portfolio as fallback for limited data scenarios.
        
        Returns:
            Dictionary containing simple portfolio results
        """
        # Equal weight portfolio
        weights = np.array([1/self.num_assets] * self.num_assets)
        
        portfolio_return, portfolio_vol, sharpe_ratio = self._portfolio_stats(weights)
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'method': 'equal_weight'
        }
    
    def find_optimal_portfolio_from_frontier(self, efficient_frontier_df: pd.DataFrame) -> Dict:
        """
        Find the optimal portfolio (highest Sharpe ratio) from the efficient frontier.
        
        Args:
            efficient_frontier_df (pd.DataFrame): Efficient frontier data
            
        Returns:
            Dict: Optimal portfolio information
        """
        if efficient_frontier_df.empty:
            return self.generate_simple_portfolio()
        
        # Find the portfolio with the highest Sharpe ratio
        optimal_idx = efficient_frontier_df['Sharpe'].idxmax()
        optimal_portfolio = efficient_frontier_df.loc[optimal_idx]
        
        return {
            'weights': optimal_portfolio['Weights'],
            'expected_return': optimal_portfolio['Return'],
            'volatility': optimal_portfolio['Volatility'],
            'sharpe_ratio': optimal_portfolio['Sharpe'],
            'method': 'frontier_optimal'
        } 