import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any
import numpy as np


def plot_efficient_frontier(efficient_frontier_df: pd.DataFrame, 
                          optimal_portfolio: Dict[str, float] = None) -> go.Figure:
    """
    Plot efficient frontier with optional optimal portfolio point.
    
    Args:
        efficient_frontier_df (pd.DataFrame): Efficient frontier data
        optimal_portfolio (Dict[str, float], optional): Optimal portfolio point
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Check if we have valid data
    if efficient_frontier_df.empty:
        fig.add_annotation(
            text="No efficient frontier data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title='Efficient Frontier')
        return fig
    
    # Ensure data is sorted by volatility for proper line plotting
    frontier_sorted = efficient_frontier_df.sort_values('Volatility')
    
    # Add scatter plot of portfolios
    fig.add_trace(go.Scatter(
        x=frontier_sorted['Volatility'],
        y=frontier_sorted['Return'],
        mode='markers',
        marker=dict(
            size=6,
            color=frontier_sorted['Sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Portfolios',
        hovertemplate='Volatility: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Add efficient frontier line (connect the optimal portfolios)
    fig.add_trace(go.Scatter(
        x=frontier_sorted['Volatility'],
        y=frontier_sorted['Return'],
        mode='lines',
        line=dict(color='red', width=3),
        name='Efficient Frontier',
        hovertemplate='Volatility: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>'
    ))
    
    # Add optimal portfolio point if provided
    if optimal_portfolio and 'volatility' in optimal_portfolio and 'expected_return' in optimal_portfolio:
        # Check if the optimal portfolio is on the frontier
        opt_vol = optimal_portfolio['volatility']
        opt_ret = optimal_portfolio['expected_return']
        
        # Find the closest point on the frontier
        distances = np.sqrt((frontier_sorted['Volatility'] - opt_vol)**2 + (frontier_sorted['Return'] - opt_ret)**2)
        closest_idx = distances.idxmin()
        closest_point = frontier_sorted.loc[closest_idx]
        
        # Add the optimal portfolio point
        fig.add_trace(go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode='markers',
            marker=dict(
                size=15,
                color='gold',
                symbol='star',
                line=dict(color='black', width=2)
            ),
            name='Optimal Portfolio',
            hovertemplate='Optimal Portfolio<br>Volatility: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: ' + f"{optimal_portfolio.get('sharpe_ratio', 0):.3f}" + '<extra></extra>'
        ))
        
        # Add annotation if the optimal portfolio is far from the frontier
        distance = np.sqrt((opt_vol - closest_point['Volatility'])**2 + (opt_ret - closest_point['Return'])**2)
        if distance > 0.05:  # If more than 5% away from frontier
            fig.add_annotation(
                x=opt_vol,
                y=opt_ret,
                text="⚠️ Suboptimal<br>Consider rebalancing",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=20,
                ay=-30,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
    
    fig.update_layout(
        title='Efficient Frontier - Optimal portfolios for each risk level',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        showlegend=True,
        hovermode='closest',
        xaxis=dict(tickformat='.3f'),
        yaxis=dict(tickformat='.3f')
    )
    
    return fig

def plot_portfolio_allocation(weights: np.ndarray, sector_info: Dict[str, str], symbols: List[str]) -> go.Figure:
    """
    Create a pie chart showing portfolio allocation.
    
    Args:
        weights: Array of portfolio weights
        sector_info: Dictionary mapping symbols to sectors
        symbols: List of stock symbols in the same order as weights
    
    Returns:
        Plotly figure object
    """
    # Ensure weights and symbols have the same length
    if len(weights) != len(symbols):
        # If lengths don't match, we need to handle this properly
        # This can happen when some stocks are removed during processing
        if len(weights) < len(symbols):
            # Some symbols were removed, use only the symbols that have weights
            symbols = symbols[:len(weights)]
        else:
            # More weights than symbols, truncate weights
            weights = weights[:len(symbols)]
    
    # Filter out zero weights to avoid cluttering the chart
    non_zero_indices = weights > 0.001  # 0.1% threshold
    filtered_weights = weights[non_zero_indices]
    filtered_symbols = [symbols[i] for i in range(len(symbols)) if non_zero_indices[i]]
    
    if len(filtered_weights) == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No portfolio allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title='Portfolio Allocation')
        return fig
    
    # Create DataFrame from weights and symbols
    weights_df = pd.DataFrame({
        'Symbol': filtered_symbols,
        'Weight': filtered_weights
    })
    
    # Add sector information
    weights_df['Sector'] = weights_df['Symbol'].map(sector_info).fillna('Unknown')
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=weights_df['Symbol'],
        values=weights_df['Weight'],
        hole=.3,
        textinfo='label+percent',
        insidetextorientation='radial',
        hovertemplate='%{label}<br>Weight: %{value:.2%}<br>Sector: %{customdata}<extra></extra>',
        customdata=weights_df['Sector']
    )])
    
    fig.update_layout(
        title='Portfolio Allocation',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_risk_metrics(risk_metrics: Dict[str, float]) -> go.Figure:
    """
    Plot risk metrics.
    
    Args:
        risk_metrics (Dict[str, float]): Risk metrics
        
    Returns:
        go.Figure: Plotly figure
    """
    # Filter out non-numeric metrics and sector concentration (handled separately)
    numeric_metrics = {}
    for key, value in risk_metrics.items():
        if key == 'liquidity_risk' and (value == 'N/A' or value is None or (isinstance(value, float) and np.isnan(value))):
            numeric_metrics['Liquidity Risk'] = 'N/A'
        elif isinstance(value, (int, float)) and key != 'sector_concentration':
            if key == 'var_95':
                numeric_metrics['VaR (95%)'] = value
            elif key == 'cvar_95':
                numeric_metrics['CVaR (95%)'] = value
            elif key == 'max_drawdown':
                numeric_metrics['Max Drawdown'] = value
            elif key == 'volatility':
                numeric_metrics['Volatility'] = value
            elif key == 'liquidity_risk':
                numeric_metrics['Liquidity Risk'] = value
            elif key == 'currency_risk':
                numeric_metrics['Currency Risk'] = value
            else:
                numeric_metrics[key] = value
    
    if not numeric_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No risk metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title='Portfolio Risk Metrics')
        return fig
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame([
        {'Metric': metric, 'Value': value} 
        for metric, value in numeric_metrics.items()
    ])
    
    # Show 'N/A' for liquidity risk
    if 'Liquidity Risk' in metrics_df['Metric'].values and (
        metrics_df.loc[metrics_df['Metric'] == 'Liquidity Risk', 'Value'].iloc[0] == 'N/A'):
        metrics_df['Value'] = metrics_df['Value'].replace('N/A', np.nan)
    
    # Create bar chart
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        title='Portfolio Risk Metrics',
        color='Metric',
        labels={'Value': 'Value', 'Metric': 'Risk Metric'}
    )
    
    # Format y-axis based on metric type
    percentage_metrics = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Volatility']
    if any(metric in numeric_metrics for metric in percentage_metrics):
        fig.update_layout(yaxis_tickformat='.1%')
    
    # Add annotation for N/A
    if 'Liquidity Risk' in metrics_df['Metric'].values and metrics_df['Value'].isnull().any():
        fig.add_annotation(
            text="Liquidity data unavailable",
            x='Liquidity Risk',
            y=0.5,
            showarrow=False,
            font=dict(color="red", size=12)
        )
    
    return fig

def plot_sector_concentration(sector_weights: Dict[str, float]) -> go.Figure:
    """
    Plot sector concentration as a bar chart.
    
    Args:
        sector_weights (Dict[str, float]): Dictionary of sector weights
        
    Returns:
        go.Figure: Plotly figure
    """
    if not sector_weights:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No sector data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title='Sector Concentration')
        return fig
    
    # Convert weights to percentages and ensure they sum to 100%
    total_weight = sum(sector_weights.values())
    if total_weight > 0:
        sector_percentages = {sector: (weight / total_weight) * 100 for sector, weight in sector_weights.items()}
    else:
        sector_percentages = sector_weights
    
    # Sort sectors by weight for better visualization
    sorted_sectors = sorted(sector_percentages.items(), key=lambda x: x[1], reverse=True)
    sectors, weights = zip(*sorted_sectors) if sorted_sectors else ([], [])
    
    fig = go.Figure(data=[
        go.Bar(
            x=sectors,
            y=weights,
            marker_color=['red' if w > 25 else 'lightblue' for w in weights],
            text=[f'{w:.1f}%' for w in weights],
            textposition='auto',
            hovertemplate='Sector: %{x}<br>Weight: %{y:.1f}%<extra></extra>'
        )
    ])
    
    # Add horizontal line for maximum recommended sector weight (50%)
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        annotation_text="50% Max Recommended",
        annotation_position="top right"
    )
    
    # Add warning annotation if any sector >25%
    if any(w > 25 for w in weights):
        for i, w in enumerate(weights):
            if w > 25:
                fig.add_annotation(
                    x=sectors[i],
                    y=w,
                    text="High!",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-30,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
    
    fig.update_layout(
        title='Sector Concentration',
        xaxis_title='Sector',
        yaxis_title='Weight (%)',
        yaxis=dict(range=[0, max(weights) * 1.1 if weights else 100]),
        showlegend=False
    )
    
    return fig 