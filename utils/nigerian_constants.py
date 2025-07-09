"""
Nigerian Portfolio Optimization Constants

This module contains all constants specific to the Nigerian market and portfolio optimization
tool, including API settings, optimization parameters, risk metrics, and UI defaults.
"""

import os
from typing import Dict, List, Tuple
from .nigerian_sectors import NIGERIAN_STOCK_SECTORS

# API Configuration
EODHD_API_KEY = os.getenv('EODHD_API_KEY', '')
EODHD_BASE_URL = "https://eodhd.com/api"

# Market Configuration

# Optimization Constraints
# OPTIMIZATION_CONSTRAINTS = {
#     'min_weight': 0.01,  # 1% minimum allocation
#     'max_weight': 0.40,  # 40% maximum allocation per stock
#     'max_sector_weight': 0.50,  # 50% maximum allocation per sector
#     'target_return': 0.20,  # 20% target annual return
#     'risk_tolerance_levels': {
#         'Conservative': 0.15,
#         'Moderate': 0.25,
#         'Aggressive': 0.35
#     }
# }

MIN_PRICE_HISTORY_DAYS = 252  # At least 1 year of data
DEFAULT_RISK_FREE_RATE = 0.1884  # 18.84% - updated to June 2025 364-day T-bill auction

# Risk Metrics Configuration
RISK_METRICS = {
    'var_confidence_level': 0.95,  # 95% confidence level for VaR
    'cvar_confidence_level': 0.95,  # 95% confidence level for CVaR
    # 'liquidity_threshold': 1000000,  # 1M NGN minimum daily volume
    # 'volatility_lookback': 252,  # 1 year for volatility calculation
    # 'correlation_threshold': 0.7,  # High correlation threshold
    # 'max_drawdown_threshold': 0.20  # 20% maximum acceptable drawdown
}

# Nigerian Market Specific Constants
# CURRENCY = "NGN"
# MARKET_HOURS = "09:00-15:00 WAT"
# SETTLEMENT_PERIOD = "T+3"  # 3 business days settlement

# Sector Information - Import from nigerian_sectors.py
# SECTORS = list(set(NIGERIAN_STOCK_SECTORS.values()))

# Cache Configuration
# CACHE_DURATION = 3600  # 1 hour in seconds
# CACHE_DIR = "cache"

# Data Validation Rules
# VALIDATION_RULES = {
#     'min_price': 0.01,  # Minimum stock price in NGN
#     'max_price': 10000,  # Maximum stock price in NGN
#     'min_volume': 1000,  # Minimum daily volume
#     'min_market_cap': 1000000000,  # 1B NGN minimum market cap
#     'max_missing_data': 0.1  # Maximum 10% missing data allowed
# }

# UI Configuration
# UI_CONFIG = {
#     'max_stocks_display': 50,
#     'default_portfolio_size': 10,
#     'chart_height': 400,
#     'table_page_size': 10
# }

# Error Messages
# ERROR_MESSAGES = {
#     'api_key_missing': "EODHD API key not found. Please set EODHD_API_KEY environment variable.",
#     'insufficient_data': "Insufficient historical data for portfolio optimization.",
#     'no_stocks_selected': "Please select at least 2 stocks for portfolio optimization.",
#     'optimization_failed': "Portfolio optimization failed. Please try different parameters.",
#     'invalid_risk_level': "Invalid risk level selected.",
#     'cache_error': "Error accessing cache. Using fresh data."
# }

# Success Messages
# SUCCESS_MESSAGES = {
#     'optimization_complete': "Portfolio optimization completed successfully.",
#     'data_loaded': "Market data loaded successfully.",
#     'cache_updated': "Cache updated with latest data."
# }

# Chart Colors
# CHART_COLORS = {
#     'primary': '#1f77b4',
#     'secondary': '#ff7f0e',
#     'success': '#2ca02c',
#     'warning': '#d62728',
#     'info': '#9467bd',
#     'light': '#8c564b',
#     'dark': '#e377c2',
#     'accent': '#7f7f7f',
#     'neutral': '#bcbd22',
#     'highlight': '#17becf'
# }

# # Portfolio Recommendations
# PORTFOLIO_RECOMMENDATIONS = {
#     'conservative': {
#         'description': 'Conservative portfolio with focus on stability and income',
#         'risk_level': 'Low',
#         'expected_return': '8-12%',
#         'suitable_for': 'Retirees, risk-averse investors'
#     },
#     'moderate': {
#         'description': 'Balanced portfolio with growth and income focus',
#         'risk_level': 'Medium',
#         'expected_return': '12-18%',
#         'suitable_for': 'Most retail investors, medium-term goals'
#     },
#     'aggressive': {
#         'description': 'Growth-focused portfolio with higher risk tolerance',
#         'risk_level': 'High',
#         'expected_return': '18-25%',
#         'suitable_for': 'Young investors, long-term growth seekers'
#     }
# }

# # Nigerian Market Characteristics
# MARKET_CHARACTERISTICS = {
#     'trading_hours': '09:30 - 14:30 WAT',
#     'trading_days': 'Monday - Friday',
#     'currency': 'Nigerian Naira (NGN)',
#     'market_cap_category': 'Frontier Market',
#     'liquidity_profile': 'Moderate to Low',
#     'volatility_profile': 'High',
#     'correlation_with_global': 'Low to Moderate'
# }

# # Regulatory Considerations
# REGULATORY_FACTORS = {
#     'sec_oversight': 'Securities and Exchange Commission (SEC)',
#     'ngx_regulation': 'Nigerian Exchange Group (NGX)',
#     'tax_implications': 'Capital gains tax on securities',
#     'foreign_investment': 'Foreign exchange controls apply',
#     'reporting_requirements': 'Quarterly financial reporting required'
# } 