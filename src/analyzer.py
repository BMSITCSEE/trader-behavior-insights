import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class TraderAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def analyze_trader_performance(self, df):
        """Analyze overall trader performance metrics"""
        trader_metrics = df.groupby('account').agg({
            'closedPnL': ['sum', 'mean', 'std', 'count'],
            'is_profitable': 'mean',
            'leverage': 'mean',
            'trade_value': 'mean'
        }).round(2)
        
        trader_metrics.columns = ['total_pnl', 'avg_pnl', 'pnl_std', 
                                 'trade_count', 'win_rate', 'avg_leverage', 
                                 'avg_trade_value']
        
        return trader_metrics
    
    def sentiment_impact_analysis(self, df):
        """Analyze trading performance under different sentiment conditions"""
        sentiment_analysis = df.groupby('classification').agg({
            'closedPnL': ['mean', 'std', 'sum'],
            'is_profitable': 'mean',
            'leverage': 'mean',
            'size': 'mean'
        }).round(2)
        
        return sentiment_analysis
    
    def trader_clustering(self, trader_metrics):
        """Cluster traders based on their behavior patterns"""
        features = ['total_pnl', 'win_rate', 'avg_leverage', 'trade_count']
        X = trader_metrics[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        trader_metrics['cluster'] = kmeans.fit_predict(X_scaled)
        
        return trader_metrics
    
    def statistical_tests(self, df):
        """Perform statistical tests on sentiment vs performance"""
        fear_pnl = df[df['classification'] == 'Fear']['closedPnL'].dropna()
        greed_pnl = df[df['classification'] == 'Greed']['closedPnL'].dropna()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(fear_pnl, greed_pnl)
        
        # Calculate effect size (Cohen's d)
        cohens_d = (fear_pnl.mean() - greed_pnl.mean()) / np.sqrt(
            ((fear_pnl.std() ** 2 + greed_pnl.std() ** 2) / 2)
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'fear_mean_pnl': fear_pnl.mean(),
            'greed_mean_pnl': greed_pnl.mean()
        }