import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import visualization components after environment setup
from app.components.visualizations import (
    plot_portfolio_allocation,
    plot_sector_concentration,
    plot_risk_metrics,
    plot_efficient_frontier
)

# Import other custom modules
from models.data_loader import DataLoader
from models.portfolio_optimizer import PortfolioOptimizer
from models.risk_calculator import RiskCalculator
from utils.nigerian_constants import DEFAULT_RISK_FREE_RATE

# --- UI Helper Functions ---
def explain_visualization(title, explanation):
    st.markdown(f"**{title}**: {explanation}")

def explain_risk_metric(metric):
    explanations = {
        'VaR (95%)': "Value at Risk (VaR) estimates the maximum expected annual loss at a 95% confidence level. For example, a 15% VaR means there's a 5% chance of losing more than 15% of your portfolio value in a year. This is now annualized for consistency with other metrics.",
        'CVaR (95%)': "Conditional Value at Risk (CVaR) measures the average annual loss in the worst 5% of cases, providing insight into tail risk. This is typically higher than VaR and represents the expected loss when the VaR threshold is breached. Now annualized for consistency.",
        'Max Drawdown': "Maximum Drawdown is the largest observed loss from a peak to a trough over the historical period. A 20% max drawdown means the portfolio lost 20% from its highest point before recovering.",
        'Volatility': "Annualized volatility measures the standard deviation of portfolio returns. Higher volatility indicates greater price swings and risk. Nigerian stocks typically have higher volatility than developed markets.",
        'Sector Concentration': "Shows how much of your portfolio is allocated to each sector. High concentration increases sector-specific risk. The red line indicates the 50% maximum recommended allocation per sector.",
        'Liquidity Risk': "Liquidity risk is based on the average daily trading volume of your portfolio holdings. Lower volume means higher risk, as it may be harder to buy or sell large amounts without affecting the price. (0 = most liquid, 1 = least liquid)"
    }
    return explanations.get(metric, "")

def main():
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader(os.getenv('EODHD_API_KEY'))

    st.title("Nigerian Portfolio Wizard")
    st.markdown("""
    This tool helps Nigerian investors build smarter portfolios using modern investing techniques and local market rules. It analyzes risk carefully and updates with live data from EODHD, so you’re always working with the latest market info.
    """)
    st.info("""
    **How to use:**
    1. **Choose Your Stocks**: Start by filtering stocks based on sector and selecting the ones you want in your portfolio.
    2. **Set Your Preferences**: Pick your risk level, how long you plan to invest (investment horizon), how often you want to rebalance, how much you’re investing, and the current risk-free interest rate.
    3. **Explore Your Results**: View your optimized portfolio, see which sectors you're invested in, understand the risk breakdown, and check out your implementation plan.
    4. **Get Insights**: Explore tailored recommendations and expert explanations to help you make smarter investment choices.
    """)

    # Message collectors
    errors = []
    warnings = []
    infos = []

    # --- Stock Selection ---
    st.header("Stock Selection")
    stocks_df = st.session_state.data_loader.get_nigerian_stocks()
    if stocks_df.empty:
        errors.append("Failed to fetch stock list from EODHD. Please check your API key and internet connection.")
        # Early return after message collection
        show_messages = True
    else:
        show_messages = False

    # Add sector info if missing
    if 'Sector' not in stocks_df.columns:
        stocks_df['Sector'] = 'Unknown'
    stocks_df['Sector'] = stocks_df['Sector'].fillna('Unknown')

    # Filter stocks by sector
    available_sectors = sorted(stocks_df['Sector'].unique())
    selected_sectors = st.multiselect(
        "Filter by Sector",
        options=available_sectors,
        default=available_sectors,
        help="Select sectors to include in your stock selection. Tip: Reducing sector concentration can help diversify your portfolio and lower risk."
    )
    
    filtered_stocks = stocks_df[stocks_df['Sector'].isin(selected_sectors)]
    if filtered_stocks.empty:
        warnings.append("No stocks match your sector filters. Please adjust the filters and try again.")
        show_messages = True

    # Display stock selection with more details
    st.subheader("Available Stocks")
    st.dataframe(
        filtered_stocks[['Code', 'Name', 'Sector']],
        use_container_width=True
    )

    selected_stocks = st.multiselect(
        "Select Stocks for Portfolio",
        options=filtered_stocks['Code'].tolist(),
        help="Choose stocks to include in your portfolio."
    )
    if not selected_stocks:
        warnings.append("Please select at least one stock.")
        show_messages = True

    # --- Sidebar: Portfolio Parameters ---
    st.sidebar.header("Portfolio Parameters")
    disable_rebalancing = st.sidebar.checkbox(
        "Disable Rebalancing",
        value=False,
        help="If checked, the portfolio will not be rebalanced automatically. Manual review is recommended. Rebalancing helps maintain your target allocation and manage risk over time."
    )
    if not disable_rebalancing:
        rebalancing_frequency = st.sidebar.selectbox(
            "Rebalancing Frequency",
            ["QUARTERLY", "MONTHLY", "ANNUALLY"],
            index=0,
            help="How often to rebalance your portfolio. More frequent rebalancing keeps your portfolio closer to your target allocation, but may increase transaction costs."
        )
    else:
        rebalancing_frequency = None
    st.sidebar.info("Periodic rebalancing helps maintain your target allocation. Disabling it may increase risk over time.")
    
    risk_level = st.sidebar.selectbox(
        "Select Risk Level",
        ["Low", "Medium", "High"],
        index=1,
        help="Choose your risk appetite. Low = conservative (lower risk, lower return), High = aggressive (higher risk, higher return). This affects the target return for optimization."
    )
    
    # Change investment horizon to a number input with plus/minus buttons
    investment_horizon = st.sidebar.number_input(
        "Investment Horizon (years)",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
        help="Select your investment time horizon in years. Longer horizons may allow for more aggressive strategies and can affect how you interpret risk and return."
    )
    
    investment_amount = st.sidebar.number_input(
        "Investment Amount (NGN)",
        min_value=100000,
        max_value=1000000000,
        value=1000000,
        step=100000,
        help="Total amount you wish to invest. This is used to calculate how much to allocate to each stock based on the optimized weights."
    )

    risk_free_rate_percent = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_RISK_FREE_RATE * 100,
        step=0.01,
        help="The risk-free rate is typically based on the latest Nigerian 364-day Treasury Bill rate (e.g., 18.84% as of June 2025). This rate is used in Sharpe ratio and portfolio optimization calculations. A higher risk-free rate makes it harder for risky assets to appear attractive."
    )
    risk_free_rate = risk_free_rate_percent / 100.0

    # --- Fetch Market Data ---
    with st.spinner("Fetching historical market data and calculating returns..."):
        try:
            market_data = st.session_state.data_loader.get_market_data(selected_stocks)
            if market_data is None:
                errors.append("❌ Failed to load market data.\n\n**Possible solutions:**\n- ✅ Ensure you've selected **at least two stocks** to proceed \n- 🔄 Try **choosing a different stock** in case the current one has limited data  \n- 📈 Opt for **stocks with a longer trading history** for better analytical accuracy  \n- 🔀 Experiment with **alternative stock combinations** to uncover better insights    \n- 🌐 **Verify your internet connection** is stable and active  ")
                if errors or warnings or infos:
                    with st.expander("Messages", expanded=True):
                        for msg in errors:
                            st.error(msg)
                        for msg in warnings:
                            st.warning(msg)
                        for msg in infos:
                            st.info(msg)
                return
        except Exception as e:
            errors.append(f"❌ Error loading market data: {str(e)}")
            if errors or warnings or infos:
                with st.expander("Messages", expanded=True):
                    for msg in errors:
                        st.error(msg)
                    for msg in warnings:
                        st.warning(msg)
                    for msg in infos:
                        st.info(msg)
            return
    if not market_data:
        errors.append("Failed to fetch market data. Please try again.")
        if errors or warnings or infos:
            with st.expander("Messages", expanded=True):
                for msg in errors:
                    st.error(msg)
                for msg in warnings:
                    st.warning(msg)
                for msg in infos:
                    st.info(msg)
        return

    # Synchronize selected_stocks with actual available data
    available_stocks = list(market_data['prices'].columns)
    if len(available_stocks) != len(selected_stocks):
        removed_stocks = set(selected_stocks) - set(available_stocks)
        if removed_stocks:
            warnings.append(f"⚠️ Some stocks were removed during data processing: {', '.join(removed_stocks)}")
            infos.append("This may be due to insufficient data, zero variance, or other data quality issues.")
        
        # Update selected_stocks to match available data
        selected_stocks = available_stocks
        if len(selected_stocks) < 2:
            errors.append("❌ Insufficient stocks available for portfolio optimization after data processing.")
            show_messages = True

    # Check data quality and provide feedback
    data_quality_issues = []
    
    # Check for sufficient data points
    data_days = len(market_data['prices'])
    if data_days < 30:
        warnings.append(f"Insufficient data points: {data_days} days (minimum 30 required)")
    
    # Check for stocks with extreme returns
    if 'returns' in market_data and not market_data['returns'].empty:
        extreme_returns = market_data['returns'][market_data['returns'] > 0.30]
        if not extreme_returns.empty:
            warnings.append(f"High volatility stocks: {', '.join(extreme_returns.index)}")
    
    # Check for stocks with zero variance
    zero_variance_stocks = []
    for col in market_data['prices'].columns:
        if market_data['prices'][col].var() == 0:
            zero_variance_stocks.append(col)
    
    if zero_variance_stocks:
        warnings.append(f"Stocks with no price movement: {', '.join(zero_variance_stocks)}")
    
    # Display data quality warnings
    if data_quality_issues:
        warnings.append("⚠️ **Data Quality Issues Detected:**")
        for issue in data_quality_issues:
            warnings.append(f"• {issue}")
    if data_quality_issues:
        infos.append("💡 **Note**: The system will attempt to handle these issues automatically, but results may be less reliable.")

    # Check if we have limited data and offer augmentation
    min_required_days = 60  # Minimum 3 months of data
    
    if data_days < min_required_days:
        errors.append(f"❌ Insufficient data: Only {data_days} days available. Need at least {min_required_days} days for meaningful analysis.\n\n**Solutions:**\n- Select stocks with longer trading history\n- Try different stock combinations\n- Contact EODHD support for extended data access")
        show_messages = True
    elif data_days < 200:  # Less than ~10 months of data
        warnings.append(f"⚠️ Limited data detected: Only {data_days} days of historical data available. This may affect portfolio optimization accuracy.")
        
        # Add option to augment data
        augment_data = st.checkbox(
            "Augment data using statistical methods (recommended for limited data)",
            value=True,
            help="Uses bootstrapping to create more data points for better optimization"
        )
        
        if augment_data:
            try:
                with st.spinner("Augmenting data for better optimization..."):
                    augmented_prices = st.session_state.data_loader.augment_limited_data(market_data['prices'], target_days=500)
                    if not augmented_prices.empty:
                        # Recalculate returns and covariance with augmented data
                        augmented_returns, augmented_cov = st.session_state.data_loader.calculate_returns_and_cov(augmented_prices)
                        if not augmented_returns.empty and not augmented_cov.empty:
                            market_data['prices'] = augmented_prices
                            market_data['returns'] = augmented_returns
                            market_data['covariance'] = augmented_cov
                            market_data['daily_returns'] = augmented_prices.pct_change().dropna()
                            infos.append(f"✅ Data augmented from {data_days} to {len(augmented_prices)} days")
                        else:
                            warnings.append("Data augmentation failed, using original data")
            except Exception as e:
                warnings.append(f"Data augmentation failed: {str(e)}. Using original data.")
    else:
        infos.append(f"✅ Sufficient data available: {data_days} days of historical data")

    # After augmentation, show min/max/mean returns in the UI
    if augment_data and 'aug_stats' in getattr(market_data['prices'], '__dict__', {}):
        aug_stats = market_data['prices'].aug_stats
        infos.append(f"**Augmented Data Stats:** Min daily return: {aug_stats['min']:.2%}, Max daily return: {aug_stats['max']:.2%}, Mean daily return: {aug_stats['mean']:.2%}")

    # --- Portfolio Optimization ---
    optimizer = PortfolioOptimizer(
        returns=market_data['returns'],
        covariance=market_data['covariance'],
        risk_free_rate=risk_free_rate  # User-configurable risk-free rate
    )
    
    # Create sector info for optimization constraints using synchronized stocks
    sector_info = {row['Code']: row['Sector'] for _, row in stocks_df.iterrows() if row['Code'] in selected_stocks}
    
    # Generate efficient frontier with fewer portfolios for better performance
    with st.spinner("Generating efficient frontier..."):
        try:
            efficient_frontier = optimizer.generate_efficient_frontier(num_portfolios=50, sector_info=sector_info)
            if efficient_frontier.empty:
                warnings.append("Could not generate efficient frontier. This may be due to insufficient data or optimization constraints.")
                optimization_result = optimizer.optimize(risk_level.lower(), sector_info=sector_info)
                if optimization_result.get('method') == 'equal_weight_fallback':
                    infos.append("⚠️ Using equal-weight portfolio due to optimization constraints.")
            else:
                # Use the optimal portfolio from the efficient frontier
                optimization_result = optimizer.find_optimal_portfolio_from_frontier(efficient_frontier)
                infos.append(f"✅ Found optimal portfolio with Sharpe ratio: {optimization_result['sharpe_ratio']:.3f}")
        except Exception as e:
            errors.append(f"Error generating efficient frontier: {str(e)}")
            show_messages = True
            efficient_frontier = pd.DataFrame()
            # Fallback to simple optimization
            optimization_result = optimizer.optimize(risk_level.lower(), sector_info=sector_info)
            if optimization_result.get('method') == 'equal_weight_fallback':
                infos.append("⚠️ Using equal-weight portfolio due to optimization failure.")

    # --- Risk Analysis ---
    risk_calculator = RiskCalculator(market_data['prices'], market_data['daily_returns'])
    
    # Validate optimization result
    if not optimization_result or 'weights' not in optimization_result:
        errors.append("❌ Portfolio optimization failed. Please try different parameters or stock selection.")
        show_messages = True
    
    if len(optimization_result['weights']) == 0:
        errors.append("❌ No portfolio weights generated. This may indicate an optimization issue.")
        show_messages = True
    
    # Add debugging information
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Portfolio weights: {optimization_result['weights']}")
    st.sidebar.write(f"Selected stocks: {selected_stocks}")
    st.sidebar.write(f"Weights length: {len(optimization_result['weights'])}")
    st.sidebar.write(f"Stocks length: {len(selected_stocks)}")
    st.sidebar.write(f"Daily returns shape: {market_data['daily_returns'].shape}")
    st.sidebar.write(f"Prices shape: {market_data['prices'].shape}")
    
    # Check for mismatch between weights and selected_stocks
    if len(optimization_result['weights']) != len(selected_stocks):
        warnings.append(f"⚠️ Mismatch detected: {len(optimization_result['weights'])} weights vs {len(selected_stocks)} stocks")
        warnings.append("This may indicate some stocks were removed during data processing.")
    
    # Add data quality validation
    if 'returns' in market_data and not market_data['returns'].empty:
        max_return = market_data['returns'].max()
        min_return = market_data['returns'].min()
        
        # Check for unrealistic returns
        if max_return > 0.50:  # More than 50% annual return
            warnings.append(f"⚠️ **Data Quality Alert**: Unrealistic returns detected (max: {max_return:.1%}). Returns have been capped at 50% for analysis.")
        
        # Check for very negative returns
        if min_return < -0.30:  # More than -30% annual return
            warnings.append(f"⚠️ **Data Quality Alert**: Very negative returns detected (min: {min_return:.1%}). This may indicate data quality issues.")
        
        # Check for zero returns
        zero_return_stocks = market_data['returns'][market_data['returns'] == 0]
        if not zero_return_stocks.empty:
            warnings.append(f"⚠️ **Data Quality Alert**: Stocks with zero returns detected: {', '.join(zero_return_stocks.index)}. These may have insufficient price movement.")
    
    # Check covariance matrix quality
    if 'covariance' in market_data and not market_data['covariance'].empty:
        cov_condition = np.linalg.cond(market_data['covariance'])
        if cov_condition > 1e6:  # High condition number indicates numerical instability
            warnings.append(f"⚠️ **Data Quality Alert**: Covariance matrix shows numerical instability (condition number: {cov_condition:.0e}). Regularization has been applied.")
    
    risk_metrics = risk_calculator.get_risk_metrics(
        optimization_result['weights'],
        sector_info=sector_info,
        volume_data=market_data.get('volume_data')
    )
    sector_weights = risk_calculator.calculate_sector_concentration(optimization_result['weights'], sector_info)

    # Sector concentration suggestion if any sector >25%
    if sector_weights and max(sector_weights.values()) > 0.25:
        top_sector = max(sector_weights, key=sector_weights.get)
        warnings.append(f"⚠️ **Diversification Suggestion:** Consider reducing exposure to {top_sector} (currently {sector_weights[top_sector]*100:.1f}%) for better diversification. Try adjusting max sector weight in constraints.")

    # --- Display Results ---
    st.header("Results & Analysis")
    
    # Display portfolio summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Expected Annual Return",
            f"{optimization_result['expected_return']:.2%}",
            f"{(optimization_result['expected_return'] * investment_amount):,.0f} NGN"
        )
    with col2:
        st.metric(
            "Portfolio Volatility",
            f"{optimization_result['volatility']:.2%}",
            "Annual Risk"
        )
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{optimization_result['sharpe_ratio']:.3f}",
            "Risk-Adjusted Return"
        )
    
    # Display investment horizon information
    st.info(f"📅 **Investment Horizon**: {investment_horizon} - Returns and risk metrics are calculated based on this time period. Longer horizons may show different risk-return profiles.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Allocation")
        explain_visualization("Portfolio Allocation", "Shows the percentage of your investment allocated to each selected stock.")
        st.plotly_chart(plot_portfolio_allocation(
            weights=optimization_result['weights'],
            sector_info=sector_info,
            symbols=selected_stocks
        ), use_container_width=True)
        st.caption("Diversification across stocks and sectors helps reduce risk.")

        st.subheader("Sector Concentration")
        explain_visualization("Sector Concentration", "Displays how your portfolio is distributed across different sectors. High concentration in one sector increases risk.")
        st.plotly_chart(plot_sector_concentration(sector_weights), use_container_width=True)

    with col2:
        st.subheader("Risk Analysis")
        explain_visualization("Risk Metrics", "Key risk metrics for your portfolio, including Value at Risk (VaR), Conditional VaR (CVaR), Maximum Drawdown, and Liquidity Risk.")
        st.plotly_chart(plot_risk_metrics(risk_metrics), use_container_width=True)
        for metric, value in risk_metrics.items():
            st.caption(f"**{metric}**: {explain_risk_metric(metric)}")
        
        # Note about liquidity risk
        st.info("💡 **Note**: Liquidity risk calculation requires volume data which is not currently available from the API. Consider checking trading volumes manually for your selected stocks.")

        st.subheader("Efficient Frontier")
        explain_visualization("Efficient Frontier", "Shows the set of optimal portfolios that offer the highest expected return for each level of risk. The red line represents the efficient frontier - any portfolio on this line is optimal for its risk level.")
        
        # Add specific explanation about the optimal portfolio
        st.markdown("""
        **Understanding the Efficient Frontier:**
        - **Red Line**: The efficient frontier - optimal portfolios for each risk level
        - **Gold Star**: Your optimal portfolio (highest Sharpe ratio)
        - **Colored Points**: All possible portfolios, colored by Sharpe ratio
        - **Goal**: The star should be ON the red line (optimal efficiency)
        """)
        
        # Debug information
        if not efficient_frontier.empty:
            infos.append(f"Generated {len(efficient_frontier)} efficient frontier portfolios")
            infos.append(f"Risk range: {efficient_frontier['Volatility'].min():.3f} - {efficient_frontier['Volatility'].max():.3f}")
            infos.append(f"Return range: {efficient_frontier['Return'].min():.3f} - {efficient_frontier['Return'].max():.3f}")
            infos.append(f"Optimal Sharpe ratio: {optimization_result.get('sharpe_ratio', 0):.3f}")
        else:
            infos.append("No efficient frontier data available")
            
        st.plotly_chart(plot_efficient_frontier(efficient_frontier, optimization_result), use_container_width=True)

    st.subheader("Implementation Plan")
    explain_visualization("Implementation Plan", "Breakdown of how much to invest in each stock based on your total investment amount.")
    
    # Add investment horizon context
    st.info(f"💰 **Investment Context**: {investment_amount:,.0f} NGN over {investment_horizon} with expected annual return of {optimization_result['expected_return']:.2%}")
    
    # Create allocation DataFrame
    # Ensure weights and selected_stocks have the same length
    if len(optimization_result['weights']) != len(selected_stocks):
        if len(optimization_result['weights']) < len(selected_stocks):
            # Some stocks were removed during processing, use only the ones with weights
            selected_stocks = selected_stocks[:len(optimization_result['weights'])]
        else:
            # More weights than stocks, truncate weights
            optimization_result['weights'] = optimization_result['weights'][:len(selected_stocks)]
    
    allocation = pd.Series(optimization_result['weights'], index=selected_stocks).mul(investment_amount).round(0)
    allocation = allocation[allocation > 0]
    
    if len(allocation) == 0:
        warnings.append("⚠️ No allocation data available. This may indicate an optimization issue.")
        show_messages = True
    
    # Create DataFrame with proper structure
    allocation_df = pd.DataFrame({
        'Symbol': allocation.index,
        'Amount (NGN)': allocation.values.astype(int)
    })
    
    # Add company names
    allocation_df['Company'] = allocation_df['Symbol'].map(stocks_df.set_index('Code')['Name'])
    
    # Reorder columns and sort by amount
    allocation_df = allocation_df[['Company', 'Symbol', 'Amount (NGN)']].sort_values('Amount (NGN)', ascending=False)
    
    # Format the display
    st.dataframe(
        allocation_df.style.format({
            'Amount (NGN)': '{:,.0f}'
        }),
        use_container_width=True
    )

    # --- General Recommendations & Insights ---
    st.header("General Recommendations & Insights")
    
    # Check for concentration risk
    max_weight = max(optimization_result['weights']) if len(optimization_result['weights']) > 0 else 0
    if max_weight > 0.3:
        warnings.append("⚠️ **High Concentration Risk Detected**: Your portfolio has a stock with over 30% allocation. Consider diversifying to reduce single-stock risk.")
    
    # Check sector concentration
    max_sector_weight = max(sector_weights.values()) if sector_weights else 0
    if max_sector_weight > 0.4:
        warnings.append("⚠️ **High Sector Concentration**: Your portfolio is heavily concentrated in one sector. Consider diversifying across sectors to reduce sector-specific risk.")
    
    st.markdown("""
    ### Key Investment Principles for Nigerian Markets:
    
    **Diversification Strategy:**
    - **Stock Level**: No single stock should exceed 25% of your portfolio
    - **Sector Level**: No single sector should exceed 50% of your portfolio
    - **Asset Classes**: Consider diversifying beyond equities (bonds, real estate, etc.)
    
    **Risk Management:**
    - **Monitor Liquidity**: Nigerian stocks can have lower trading volumes
    - **Rebalance Regularly**: Review and rebalance your portfolio quarterly
    - **Watch Transaction Costs**: High turnover increases costs in Nigerian markets
    - **Stay Informed**: Keep up with macroeconomic and political developments
    
    **Market-Specific Considerations:**
    - **Currency Risk**: Monitor NGN/USD exchange rates if you have foreign exposure
    - **Regulatory Changes**: Stay updated on SEC and NGX regulatory developments
    - **Political Risk**: Nigerian markets can be sensitive to political developments
    - **Inflation Impact**: Consider inflation's effect on real returns
    
    **Portfolio Optimization:**
    - **Efficient Frontier**: Your portfolio should lie on or near the efficient frontier
    - **Risk-Return Balance**: Higher returns typically come with higher risk
    - **Time Horizon**: Align your portfolio with your investment timeline
    - **Risk Tolerance**: Ensure your portfolio matches your risk appetite
    """)
    
    st.info("""
    **Disclaimer**: Nigerian markets can be volatile and illiquid. Always consult a licensed financial advisor 
    before making investment decisions. Past performance does not necessarily guarantee future results.
    """)

    # At the end, display all messages in one expander
    if errors or warnings or infos:
        with st.expander("Messages", expanded=True):
            for msg in errors:
                st.error(msg)
            for msg in warnings:
                st.warning(msg)
            for msg in infos:
                st.info(msg)

# Make sure main() is callable from outside
__all__ = ['main'] 