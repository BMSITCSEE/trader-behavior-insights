import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
    
    
    def clean_trader_data(self, df):

        # Standardize column names
        df = df.rename(columns={
        'Account': 'account',
        'Side': 'side',
        'Coin': 'symbol',  # Assuming Coin = symbol
        })

        """Clean and preprocess trader data"""
        # Handle missing values
        df['closedPnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce')
        df.drop(columns=['Closed PnL'], inplace=True)  # optional: rename and clean
        df['size'] = pd.to_numeric(df['Size USD'], errors='coerce')
        df.drop(columns=['Size USD'], inplace=True)  # optional cleanup
        df['execution_price'] = pd.to_numeric(df['Execution Price'], errors='coerce')
        df.drop(columns=['Execution Price'], inplace=True)  # optional cleanup
        #df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
        # Add synthetic leverage column with random float values (1x to 10x)
        np.random.seed(42)
        df['leverage'] = np.random.uniform(1, 10, size=len(df)).round(2)

        
        # Remove rows with critical missing values
        df = df.dropna(subset=['account', 'symbol', 'time', 'side'])
        
        # Fill numerical missing values
        df['closedPnL'] = df['closedPnL'].fillna(0)
        #df['leverage'] = df['leverage'].fillna(1)
        
        return df
    
    def create_features(self, df):
        """Create additional features for analysis"""
        # Time-based features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Trading features
        df['trade_value'] = df['size'] * df['execution_price']
        df['is_profitable'] = (df['closedPnL'] > 0).astype(int)
        
        # Calculate cumulative metrics per account
        df = df.sort_values(['account', 'time'])
        df['cumulative_pnl'] = df.groupby('account')['closedPnL'].cumsum()
        df['trade_count'] = df.groupby('account').cumcount() + 1
        
        return df