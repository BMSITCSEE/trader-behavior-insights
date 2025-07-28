# Trader Behavior Insights: Market Sentiment Analysis

## Project Overview
This project analyzes the relationship between trader performance on Hyperliquid and Bitcoin market sentiment (Fear/Greed Index) to uncover hidden patterns and deliver actionable trading insights.

## Key Findings
- **Statistical Analysis**: Found statistically significant differences in trader performance between Fear and Greed market periods (p < 0.05)
- **Trader Segmentation**: Identified 4 distinct trader types using machine learning clustering
- **Optimal Trading Conditions**: Discovered best trading hours and risk parameters
- **Sentiment Impact**: Quantified how market sentiment affects leverage usage and position sizing

## Methodology
1. **Data Preprocessing**: Cleaned and merged trader data with sentiment indicators
2. **Exploratory Analysis**: Identified key patterns in trader behavior
3. **Performance Metrics**: Analyzed PnL distribution across different sentiment periods
4. **Statistical Analysis**: Correlation between sentiment and trading outcomes
5. **Predictive Insights**: Machine learning models for trader success prediction

## Technologies Used
- Python 3.9+
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn, Plotly for visualization
- Scikit-learn for machine learning
- Jupyter notebooks for analysis

## Project Structure

trader-behavior-insights/
├── notebooks/ # Detailed analysis notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_sentiment_analysis.ipynb
│ ├── 03_trader_performance_analysis.ipynb
│ └── 04_insights_and_visualization.ipynb
├── src/ # Modular source code
│ ├── data_loader.py # Data loading utilities
│ ├── preprocessor.py # Data preprocessing
│ ├── analyzer.py # Statistical analysis
│ └── visualizer.py # Visualization functions
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Processed data
├── results/
│ ├── figures/ # 13 visualizations
│ └── reports/ # 11 analysis reports
├── main.py # Main analysis pipeline
├── generate_all_reports.py # Report generation
└── debug_and_generate_all_visuals.py # Visualization generation

## Installation & Setup

Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

Step 1: Clone the Repository
```bash
git clone https://github.com/BMSITCSEE/trader-behavior-insights.git
cd trader-behavior-insights

Step 2: Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Download Datasets
Download the datasets from the provided Google Drive links
Place them in the data/raw/ directory:
Rename historical data to historical_trader_data.csv
Rename sentiment data to fear_greed_index.csv

Step 5: Run the Analysis
python main.py

This will:
1.Process and clean the data
2.Perform statistical analysis
3.Generate machine learning models
4.Create all visualizations and reports

Analysis Components
1. Data Exploration (notebooks/01_data_exploration.ipynb)
Dataset overview and quality assessment
Missing value analysis
Initial pattern identification

2. Sentiment Analysis (notebooks/02_sentiment_analysis.ipynb)
Fear vs Greed performance comparison
Statistical significance testing (t-tests, effect size)
Risk metrics by sentiment

3. Trader Performance Analysis (notebooks/03_trader_performance_analysis.ipynb)
Individual trader metrics
K-means clustering for trader segmentation
Performance evolution over time

4. Insights & Visualization (notebooks/04_insights_and_visualization.ipynb)
Executive dashboard
Interactive visualizations
Actionable recommendations

Key Deliverables
Visualizations (13 files in results/figures/)
pnl_distribution.png: PnL distribution by market sentiment
trader_clusters.html: Interactive 3D trader segmentation
time_analysis.html: Time-based trading patterns
executive_dashboard.html: Comprehensive overview dashboard
sentiment_comparison.png: Fear vs Greed metrics comparison
risk_metrics_sentiment.png: Risk analysis by sentiment
And 7 more detailed visualizations

Reports (11 files in results/reports/)
FINAL_REPORT.md: Comprehensive analysis summary
trader_performance_metrics.csv: Detailed trader statistics
sentiment_insights.json: Key sentiment findings
recommendations.json: Actionable trading strategies
analysis_report.md: Technical analysis details
And 6 more analytical reports

Key Insights
Sentiment Impact: Traders show significantly different behavior during Fear vs Greed periods
Trader Segments:
Conservative Traders: Low leverage, steady returns
High Volume Traders: Frequent trading, market makers
Risk Takers: High leverage, volatile returns
Professional Traders: Balanced approach, best risk-adjusted returns
Optimal Conditions: Best trading occurs during specific hours with moderate leverage
Risk Management: Position sizing should adapt to market sentiment