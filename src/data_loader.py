import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, trader_data_path, sentiment_data_path):
        self.trader_data_path = trader_data_path
        self.sentiment_data_path = sentiment_data_path
        
    def load_trader_data(self):
        """Load historical trader data from Hyperliquid"""
        df = pd.read_csv(self.trader_data_path)
        df['time'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.drop(columns=['Timestamp'], inplace=True)
        return df
    
    def load_sentiment_data(self):
        """Load Bitcoin Fear/Greed Index data"""
        df = pd.read_csv(self.sentiment_data_path)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def merge_datasets(self, trader_df, sentiment_df):
        """Merge trader and sentiment data based on date"""
        trader_df['date'] = trader_df['time'].dt.date
        sentiment_df['date'] = sentiment_df['Date'].dt.date
        
        merged_df = pd.merge(
            trader_df, 
            sentiment_df[['date', 'classification']], 
            on='date', 
            how='left'
        )
        return merged_df