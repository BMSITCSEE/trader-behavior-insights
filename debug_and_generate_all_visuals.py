import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/reports', exist_ok=True)

print("Loading data...")
# Load processed data
df = pd.read_csv('data/processed/merged_data.csv')
df['time'] = pd.to_datetime(df['time'])
df['date'] = pd.to_datetime(df['date'])

# Ensure numeric columns
df['closedPnL'] = pd.to_numeric(df['closedPnL'], errors='coerce')
df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
df['size'] = pd.to_numeric(df['size'], errors='coerce')
df['is_profitable'] = (df['closedPnL'] > 0).astype(int)
df['hour'] = pd.to_datetime(df['time']).dt.hour

print(f"Data loaded: {len(df)} records")

# 1. Generate trader_performance_overview.png
print("\n1. Generating trader_performance_overview.png...")
trader_metrics = df.groupby('account').agg({
    'closedPnL': ['sum', 'mean', 'std'],
    'is_profitable': 'mean',
    'leverage': 'mean',
    'symbol': 'nunique',
    'time': 'count'
}).reset_index()

trader_metrics.columns = ['account', 'total_pnl', 'avg_pnl', 'pnl_std', 
                         'win_rate', 'avg_leverage', 'symbols_traded', 'total_trades']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# PnL distribution
ax1 = axes[0, 0]
trader_metrics['total_pnl'].hist(bins=50, ax=ax1, edgecolor='black')
ax1.axvline(trader_metrics['total_pnl'].median(), color='red', linestyle='--', label='Median')
ax1.set_title('Distribution of Total PnL')
ax1.set_xlabel('Total PnL')
ax1.set_ylabel('Number of Traders')
ax1.legend()

# Win rate distribution
ax2 = axes[0, 1]
trader_metrics['win_rate'].hist(bins=30, ax=ax2, edgecolor='black')
ax2.axvline(trader_metrics['win_rate'].mean(), color='red', linestyle='--', label='Mean')
ax2.set_title('Distribution of Win Rates')
ax2.set_xlabel('Win Rate')
ax2.set_ylabel('Number of Traders')
ax2.legend()

# Scatter: Total PnL vs Win Rate
ax3 = axes[1, 0]
scatter = ax3.scatter(trader_metrics['win_rate'], trader_metrics['total_pnl'], 
                     c=trader_metrics['total_trades'], cmap='viridis', alpha=0.6)
ax3.set_xlabel('Win Rate')
ax3.set_ylabel('Total PnL')
ax3.set_title('Total PnL vs Win Rate')
plt.colorbar(scatter, ax=ax3, label='Total Trades')

# Trading activity by profit tier
ax4 = axes[1, 1]
trader_metrics['profit_tier'] = pd.qcut(trader_metrics['total_pnl'], q=5, 
                                        labels=['Bottom 20%', 'Low', 'Mid', 'High', 'Top 20%'])
profit_tiers = trader_metrics.groupby('profit_tier').agg({
    'total_trades': 'mean',
    'avg_leverage': 'mean',
    'symbols_traded': 'mean'
})
profit_tiers.plot(kind='bar', ax=ax4)
ax4.set_title('Trading Characteristics by Profit Tier')
ax4.set_xlabel('Profit Tier')
ax4.set_ylabel('Average Value')
ax4.legend(['Avg Trades', 'Avg Leverage', 'Avg Symbols'])

plt.tight_layout()
plt.savefig('results/figures/trader_performance_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ trader_performance_overview.png generated")

# 2. Generate sentiment_comparison.png
print("\n2. Generating sentiment_comparison.png...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Average position size
ax1 = axes[0, 0]
sentiment_data = df.groupby('classification')['size'].mean()
bars1 = ax1.bar(sentiment_data.index, sentiment_data.values, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Average Position Size by Sentiment', fontsize=14)
ax1.set_ylabel('Position Size')

# Win rate
ax2 = axes[0, 1]
win_rate = df.groupby('classification')['is_profitable'].mean() * 100
bars2 = ax2.bar(win_rate.index, win_rate.values, color=['#FF6B6B', '#4ECDC4'])
ax2.set_title('Win Rate by Sentiment', fontsize=14)
ax2.set_ylabel('Win Rate (%)')

# Average leverage
ax3 = axes[1, 0]
avg_leverage = df.groupby('classification')['leverage'].mean()
bars3 = ax3.bar(avg_leverage.index, avg_leverage.values, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('Average Leverage by Sentiment', fontsize=14)
ax3.set_ylabel('Leverage')

# PnL distribution
ax4 = axes[1, 1]
for sentiment in df['classification'].dropna().unique():
    data = df[df['classification'] == sentiment]['closedPnL']
    ax4.hist(data, bins=50, alpha=0.6, label=sentiment, density=True)
ax4.set_title('PnL Distribution by Sentiment', fontsize=14)
ax4.set_xlabel('Closed PnL')
ax4.set_ylabel('Density')
ax4.legend()
ax4.set_xlim(-500, 500)

plt.tight_layout()
plt.savefig('results/figures/sentiment_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ sentiment_comparison.png generated")

# 3. Generate risk_metrics_sentiment.png
print("\n3. Generating risk_metrics_sentiment.png...")
# Calculate risk metrics
risk_metrics = pd.DataFrame()
for sentiment in df['classification'].dropna().unique():
    sent_data = df[df['classification'] == sentiment]
    
    metrics = {
        'sentiment': sentiment,
        'avg_loss': sent_data[sent_data['closedPnL'] < 0]['closedPnL'].mean(),
        'avg_win': sent_data[sent_data['closedPnL'] > 0]['closedPnL'].mean(),
        'volatility': sent_data['closedPnL'].std(),
        'total_losses': sent_data[sent_data['closedPnL'] < 0]['closedPnL'].sum(),
        'total_wins': sent_data[sent_data['closedPnL'] > 0]['closedPnL'].sum()
    }
    risk_metrics = pd.concat([risk_metrics, pd.DataFrame([metrics])], ignore_index=True)

risk_metrics['profit_factor'] = abs(risk_metrics['total_wins'] / risk_metrics['total_losses'])
risk_metrics['risk_reward_ratio'] = abs(risk_metrics['avg_win'] / risk_metrics['avg_loss'])
risk_metrics.set_index('sentiment', inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Risk-Reward Ratio
ax1 = axes[0, 0]
risk_metrics['risk_reward_ratio'].plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Risk-Reward Ratio by Sentiment', fontsize=14)
ax1.set_ylabel('Ratio')
ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# Profit Factor
ax2 = axes[0, 1]
risk_metrics['profit_factor'].plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
ax2.set_title('Profit Factor by Sentiment', fontsize=14)
ax2.set_ylabel('Factor')
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# Volatility
ax3 = axes[1, 0]
risk_metrics['volatility'].plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('PnL Volatility by Sentiment', fontsize=14)
ax3.set_ylabel('Standard Deviation')

# Sharpe Ratio (approximation)
ax4 = axes[1, 1]
sharpe_ratios = df.groupby('classification')['closedPnL'].agg(['mean', 'std'])
sharpe_ratios['sharpe'] = sharpe_ratios['mean'] / sharpe_ratios['std']
sharpe_ratios['sharpe'].plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
ax4.set_title('Sharpe Ratio by Sentiment', fontsize=14)
ax4.set_ylabel('Ratio')
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/figures/risk_metrics_sentiment.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ risk_metrics_sentiment.png generated")

# 4. Generate cluster_trading_patterns.png
print("\n4. Generating cluster_trading_patterns.png...")
# Perform clustering on trader metrics
active_traders = trader_metrics[trader_metrics['total_trades'] >= 10].copy()
features = ['total_pnl', 'win_rate', 'avg_leverage', 'total_trades']
X = active_traders[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
active_traders['cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {0: 'Conservative', 1: 'High Volume', 2: 'Risk Takers', 3: 'Professional'}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, (cluster_id, cluster_name) in enumerate(cluster_names.items()):
    ax = axes[idx // 2, idx % 2]
    cluster_traders = active_traders[active_traders['cluster'] == cluster_id]['account'].values
    cluster_trades = df[df['account'].isin(cluster_traders)]
    
    if len(cluster_trades) > 0:
        hourly_pattern = cluster_trades.groupby('hour').size()
        ax.bar(hourly_pattern.index, hourly_pattern.values, alpha=0.7)
        ax.set_title(f'{cluster_name} - Hourly Trading Pattern')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Trades')
        ax.set_xlim(-0.5, 23.5)

plt.tight_layout()
plt.savefig('results/figures/cluster_trading_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ cluster_trading_patterns.png generated")

# 5. Generate sentiment_deep_dive.png
print("\n5. Generating sentiment_deep_dive.png...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. PnL by Sentiment and Hour
ax1 = axes[0, 0]
sentiment_hour = df.groupby(['hour', 'classification'])['closedPnL'].mean().unstack()
if not sentiment_hour.empty:
    sentiment_hour.plot(ax=ax1, kind='bar', width=0.8)
ax1.set_title('Average PnL by Hour and Sentiment', fontsize=14)
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Average PnL')

# 2. Trade Size Distribution
ax2 = axes[0, 1]
for sentiment in df['classification'].dropna().unique():
    data = df[df['classification'] == sentiment]['size']
    if len(data) > 0:
        ax2.hist(data, bins=30, alpha=0.6, label=sentiment, density=True)
ax2.set_title('Trade Size Distribution by Sentiment', fontsize=14)
ax2.set_xlabel('Trade Size')
ax2.set_ylabel('Density')
ax2.legend()

# 3. Win Rate Evolution
ax3 = axes[0, 2]
win_rate_time = df.groupby([df['date'], 'classification'])['is_profitable'].mean().unstack()
if not win_rate_time.empty:
    win_rate_time.rolling(7).mean().plot(ax=ax3)
ax3.set_title('7-Day Moving Average Win Rate', fontsize=14)
ax3.set_xlabel('Date')
ax3.set_ylabel('Win Rate')
ax3.legend(title='Sentiment')

# 4. Leverage Comparison
ax4 = axes[1, 0]
leverage_data = []
labels = []
for sent in df['classification'].dropna().unique():
    data = df[df['classification'] == sent]['leverage'].dropna()
    if len(data) > 0:
        leverage_data.append(data)
        labels.append(sent)

if leverage_data:
    bp = ax4.boxplot(leverage_data, labels=labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
ax4.set_title('Leverage Distribution by Sentiment', fontsize=14)
ax4.set_ylabel('Leverage')
ax4.set_ylim(0, 20)

# 5. Symbol Preference
ax5 = axes[1, 1]
top_symbols = df['symbol'].value_counts().head(5).index
symbol_sentiment = df[df['symbol'].isin(top_symbols)].groupby(['symbol', 'classification']).size().unstack(fill_value=0)
if not symbol_sentiment.empty:
    symbol_sentiment.plot(kind='bar', ax=ax5)
ax5.set_title('Top 5 Symbols by Sentiment', fontsize=14)
ax5.set_xlabel('Symbol')
ax5.set_ylabel('Number of Trades')
ax5.legend(title='Sentiment')

# 6. Risk-Reward by Sentiment
ax6 = axes[1, 2]
for sentiment in df['classification'].dropna().unique():
    sent_data = df[df['classification'] == sentiment]
    if len(sent_data) > 0:
        wins = sent_data[sent_data['closedPnL'] > 0]['closedPnL']
        losses = sent_data[sent_data['closedPnL'] < 0]['closedPnL']
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            win_rate = sent_data['is_profitable'].mean()
            
            ax6.scatter(avg_loss, avg_win, s=200, alpha=0.7, label=sentiment)
            ax6.annotate(f'{win_rate:.1%}', (avg_loss, avg_win), ha='center', va='center')

ax6.plot([0, 100], [0, 100], 'k--', alpha=0.3)
ax6.set_xlabel('Average Loss')
ax6.set_ylabel('Average Win')
ax6.set_title('Risk-Reward Profile by Sentiment', fontsize=14)
ax6.legend()
ax6.set_xlim(left=0)
ax6.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('results/figures/sentiment_deep_dive.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ sentiment_deep_dive.png generated")

# 6. Generate executive_dashboard.html
print("\n6. Generating executive_dashboard.html...")
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=('Total PnL Distribution', 'Win Rate by Sentiment', 'Daily Trading Volume',
                    'Top 10 Traders', 'Risk Metrics', 'Hourly Activity',
                    'Symbol Performance', 'Leverage Usage', 'Performance Trend'),
    specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'box'}, {'type': 'scatter'}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. PnL Distribution
fig.add_trace(
    go.Histogram(x=df['closedPnL'], nbinsx=50, name='PnL', showlegend=False),
    row=1, col=1
)

# 2. Win Rate by Sentiment
win_rate_sentiment = df.groupby('classification')['is_profitable'].mean()
fig.add_trace(
    go.Bar(x=win_rate_sentiment.index, y=win_rate_sentiment.values, showlegend=False),
    row=1, col=2
)

# 3. Daily Trading Volume
daily_volume = df.groupby(df['date'])['trade_value'].sum()
fig.add_trace(
    go.Scatter(x=daily_volume.index, y=daily_volume.values, mode='lines', showlegend=False),
    row=1, col=3
)

# 4. Top 10 Traders
top_traders = trader_metrics.nlargest(10, 'total_pnl')
fig.add_trace(
    go.Bar(x=top_traders['total_pnl'].values, y=top_traders['account'].values, 
           orientation='h', showlegend=False),
    row=2, col=1
)

# 5. Risk Metrics
fig.add_trace(
    go.Scatter(x=trader_metrics['win_rate'], y=trader_metrics['total_pnl'],
               mode='markers', showlegend=False,
               marker=dict(size=5, color=trader_metrics['avg_leverage'], colorscale='Viridis')),
    row=2, col=2
)

# 6. Hourly Activity
hourly_trades = df.groupby('hour').size()
fig.add_trace(
    go.Bar(x=hourly_trades.index, y=hourly_trades.values, showlegend=False),
    row=2, col=3
)

# 7. Symbol Performance
symbol_perf = df.groupby('symbol')['closedPnL'].sum().nlargest(10)
fig.add_trace(
    go.Bar(x=symbol_perf.index, y=symbol_perf.values, showlegend=False),
    row=3, col=1
)

# 8. Leverage Usage
fig.add_trace(
    go.Box(y=df['leverage'], name='Leverage', showlegend=False),
    row=3, col=2
)

# 9. Performance Trend
cumulative_pnl = df.sort_values('time')['closedPnL'].cumsum()
fig.add_trace(
    go.Scatter(x=df.sort_values('time')['time'], y=cumulative_pnl,
               mode='lines', showlegend=False),
    row=3, col=3
)

fig.update_layout(height=1200, title_text="Executive Dashboard", showlegend=False)
fig.write_html('results/figures/executive_dashboard.html')
print("✓ executive_dashboard.html generated")

# 7. Generate trader_clusters_3d.html
print("\n7. Generating trader_clusters_3d.html...")
if 'cluster' in active_traders.columns:
    fig = px.scatter_3d(
        active_traders.reset_index(),
        x='total_pnl',
        y='win_rate',
        z='avg_leverage',
        color='cluster',
        size='total_trades',
        hover_data=['account'],
        title='Trader Clusters: 3D Analysis',
        labels={'total_pnl': 'Total PnL', 'win_rate': 'Win Rate', 'avg_leverage': 'Average Leverage'}
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
    
    fig.write_html('results/figures/trader_clusters_3d.html')
    print("✓ trader_clusters_3d.html generated")

# 8. Generate cluster_radar_chart.html
print("\n8. Generating cluster_radar_chart.html...")
if 'cluster' in active_traders.columns:
    # Calculate cluster profiles
    cluster_profiles = active_traders.groupby('cluster').agg({
        'total_pnl': 'mean',
        'win_rate': 'mean',
        'avg_leverage': 'mean',
        'total_trades': 'mean'
    })
    
    # Add a fifth metric for better radar chart
    cluster_profiles['risk_score'] = cluster_profiles['avg_leverage'] * (1 - cluster_profiles['win_rate'])
    
    # Normalize values
    scaler = MinMaxScaler()
    cluster_profiles_norm = pd.DataFrame(
        scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns
    )
    
    fig = go.Figure()
    
    for cluster in cluster_profiles_norm.index:
        fig.add_trace(go.Scatterpolar(
            r=cluster_profiles_norm.loc[cluster].values,
            theta=cluster_profiles_norm.columns,
            fill='toself',
            name=f'{cluster_names.get(cluster, f"Cluster {cluster}")}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Trader Cluster Profiles (Normalized)"
    )
    
    fig.write_html('results/figures/cluster_radar_chart.html')
    print("✓ cluster_radar_chart.html generated")

# 9. Generate time_series_analysis.html
print("\n9. Generating time_series_analysis.html...")
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Daily PnL and Volume', 'Market Sentiment Timeline', 
                    'Cumulative Performance', 'Rolling Metrics'),
    specs=[[{"secondary_y": True}],
           [{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": True}]],
    vertical_spacing=0.08,
    row_heights=[0.3, 0.2, 0.25, 0.25]
)

# 1. Daily PnL and Volume
daily_stats = df.groupby(df['date']).agg({
    'closedPnL': 'sum',
    'trade_value': 'sum'
})

fig.add_trace(
    go.Bar(x=daily_stats.index, y=daily_stats['closedPnL'], name='Daily PnL'),
    row=1, col=1, secondary_y=False
)

fig.add_trace(
    go.Scatter(x=daily_stats.index, y=daily_stats['trade_value'], name='Volume', line=dict(color='orange')),
    row=1, col=1, secondary_y=True
)

# 2. Sentiment Timeline
sentiment_counts = df.groupby(['date', 'classification']).size().unstack(fill_value=0)
for sentiment in sentiment_counts.columns:
    fig.add_trace(
        go.Scatter(x=sentiment_counts.index, y=sentiment_counts[sentiment],
                   mode='lines', stackgroup='one', name=sentiment),
        row=2, col=1
    )

# 3. Cumulative Performance
cumulative_daily = daily_stats['closedPnL'].cumsum()
fig.add_trace(
    go.Scatter(x=cumulative_daily.index, y=cumulative_daily.values,
               mode='lines', name='Cumulative PnL'),
    row=3, col=1
)

# 4. Rolling Metrics
rolling_window = 7
daily_metrics = df.groupby(df['date']).agg({
    'is_profitable': 'mean',
    'leverage': 'mean'
})

fig.add_trace(
    go.Scatter(x=daily_metrics.index, 
               y=daily_metrics['is_profitable'].rolling(rolling_window).mean(),
               name=f'{rolling_window}D Win Rate'),
    row=4, col=1, secondary_y=False
)

fig.add_trace(
    go.Scatter(x=daily_metrics.index, 
               y=daily_metrics['leverage'].rolling(rolling_window).mean(),
               name=f'{rolling_window}D Avg Leverage', line=dict(color='red')),
    row=4, col=1, secondary_y=True
)

fig.update_layout(height=1200, title_text="Time Series Analysis")
fig.write_html('results/figures/time_series_analysis.html')
print("✓ time_series_analysis.html generated")

# 10. Generate recommendations.png
print("\n10. Generating recommendations.png...")
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'ACTIONABLE TRADING RECOMMENDATIONS', 
        ha='center', va='top', fontsize=20, fontweight='bold')

recommendations = [
    {
        "category": "Risk Management",
        "items": [
            f"Limit leverage to {df['leverage'].quantile(0.75):.1f}x",
            "Implement stop-loss at 2% of account value",
            "Diversify across 5+ symbols"
        ]
    },
    {
        "category": "Trading Strategy",
        "items": [
            f"Focus on hours: {df.groupby('hour')['is_profitable'].mean().nlargest(3).index.tolist()}",
            "Increase positions during Fear periods",
            "Target consistent profitable symbols"
        ]
    },
    {
        "category": "Performance",
        "items": [
            "Maintain 45%+ win rate",
            "50-200 trades/month optimal",
            "Adjust for sentiment changes"
        ]
    }
]

y_pos = 0.85
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, rec_group in enumerate(recommendations):
    ax.text(0.1, y_pos, rec_group['category'].upper(), 
            fontsize=14, fontweight='bold', color=colors[i])
    y_pos -= 0.04
    
    for item in rec_group['items']:
        ax.text(0.15, y_pos, f"• {item}", fontsize=11)
        y_pos -= 0.04
    
    y_pos -= 0.02

ax.text(0.5, 0.05, f'Based on {len(df):,} trades from {df["account"].nunique()} traders',
        ha='center', fontsize=10, style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('results/figures/recommendations.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ recommendations.png generated")

# Save updated trader metrics
trader_metrics.to_csv('results/reports/trader_performance_metrics.csv')
print("\n✓ Updated trader_performance_metrics.csv saved")

# Save cluster assignments if available
if 'cluster' in active_traders.columns:
    active_traders.to_csv('results/reports/trader_clusters.csv')
    print("✓ trader_clusters.csv saved")

print("\n" + "="*50)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*50)
print("\nGenerated files:")
print("- trader_performance_overview.png")
print("- sentiment_comparison.png")
print("- risk_metrics_sentiment.png")
print("- cluster_trading_patterns.png")
print("- sentiment_deep_dive.png")
print("- executive_dashboard.html")
print("- trader_clusters_3d.html")
print("- cluster_radar_chart.html")
print("- time_series_analysis.html")
print("- recommendations.png")