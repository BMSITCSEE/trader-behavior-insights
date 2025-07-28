import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class Visualizer:
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']
        sns.set_style("whitegrid")
        
    def plot_pnl_distribution(self, df, save_path=None):
        """Plot PnL distribution by sentiment"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribution plot
        for sentiment in df['classification'].unique():
            if pd.notna(sentiment):
                data = df[df['classification'] == sentiment]['closedPnL']
                ax1.hist(data, bins=50, alpha=0.6, label=sentiment, density=True)
        
        ax1.set_xlabel('Closed PnL')
        ax1.set_ylabel('Density')
        ax1.set_title('PnL Distribution by Market Sentiment')
        ax1.legend()
        ax1.set_xlim(-1000, 1000)  # Adjust based on your data
        
        # Box plot
        df_clean = df[df['classification'].notna()]
        sns.boxplot(data=df_clean, x='classification', y='closedPnL', ax=ax2)
        ax2.set_title('PnL Comparison by Sentiment')
        ax2.set_ylim(-500, 500)  # Adjust based on your data
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_trader_clusters(self, trader_metrics, save_path=None):
        """Create interactive 3D scatter plot of trader clusters"""
        fig = px.scatter_3d(
            trader_metrics.reset_index(),
            x='total_pnl',
            y='win_rate',
            z='avg_leverage',
            color='cluster',
            hover_data=['account', 'trade_count'],
            title='Trader Clustering Analysis',
            labels={
                'total_pnl': 'Total PnL',
                'win_rate': 'Win Rate',
                'avg_leverage': 'Average Leverage'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Total PnL',
                yaxis_title='Win Rate',
                zaxis_title='Average Leverage'
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def plot_time_analysis(self, df, save_path=None):
        """Plot trading patterns over time"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Trading Volume', 'Daily PnL Trend',
                          'Win Rate by Day of Week', 'Leverage Usage Over Time')
        )
        
        # Hourly volume
        hourly_stats = df.groupby('hour').agg({
            'trade_value': 'sum',
            'closedPnL': 'mean'
        })
        
        fig.add_trace(
            go.Bar(x=hourly_stats.index, y=hourly_stats['trade_value'],
                  name='Trading Volume'),
            row=1, col=1
        )
        
        # Daily PnL trend
        daily_pnl = df.groupby(df['time'].dt.date)['closedPnL'].sum()
        fig.add_trace(
            go.Scatter(x=daily_pnl.index, y=daily_pnl.values,
                      mode='lines+markers', name='Daily PnL'),
            row=1, col=2
        )
        
        # Win rate by day of week
        dow_stats = df.groupby('day_of_week')['is_profitable'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(
            go.Bar(x=days, y=dow_stats.values, name='Win Rate'),
            row=2, col=1
        )
        
        # Leverage over time
        leverage_trend = df.groupby(df['time'].dt.date)['leverage'].mean()
        fig.add_trace(
            go.Scatter(x=leverage_trend.index, y=leverage_trend.values,
                      mode='lines', name='Avg Leverage'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        
        if save_path:
            fig.write_html(save_path)
        fig.show()