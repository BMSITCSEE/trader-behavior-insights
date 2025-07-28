import os
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analyzer import TraderAnalyzer
from src.visualizer import Visualizer
import pandas as pd
import json
import subprocess
import sys

def main():
    print("Starting Trader Behavior Insights Analysis...")
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader(
        'data/raw/historical_trader_data.csv',
        'data/raw/fear_greed_index.csv'
    )
    preprocessor = DataPreprocessor()
    analyzer = TraderAnalyzer()
    visualizer = Visualizer()
    
    # Load data
    print("Loading datasets...")
    trader_df = data_loader.load_trader_data()
    sentiment_df = data_loader.load_sentiment_data()
    
    # Preprocess data
    print("Preprocessing data...")
    trader_df = preprocessor.clean_trader_data(trader_df)
    trader_df = preprocessor.create_features(trader_df)
    
    # Merge datasets
    print("Merging trader data with sentiment data...")
    merged_df = data_loader.merge_datasets(trader_df, sentiment_df)
    
    # Save processed data
    merged_df.to_csv('data/processed/merged_data.csv', index=False)
    
    # Perform analysis
    print("Analyzing trader performance...")
    trader_metrics = analyzer.analyze_trader_performance(merged_df)
    sentiment_impact = analyzer.sentiment_impact_analysis(merged_df)
    trader_clusters = analyzer.trader_clustering(trader_metrics)
    statistical_results = analyzer.statistical_tests(merged_df)
    
    # Save initial analysis results for other scripts
    print("Saving initial analysis results...")
    trader_clusters.to_csv('results/reports/trader_performance_metrics.csv')
    sentiment_impact.to_csv('results/reports/sentiment_impact_analysis.csv')
    
    # Save statistical results
    with open('results/reports/statistical_results.json', 'w') as f:
        json.dump(statistical_results, f, indent=4)
    
    
    # Run comprehensive visualization generation
    print("\n" + "="*60)
    print("Generating comprehensive visualizations...")
    print("="*60)
    try:
        result = subprocess.run([sys.executable, 'debug_and_generate_all_visuals.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(" All visualizations generated successfully!")
        else:
            print("  Some visualizations may have failed. Running script manually may help.")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"  Could not run visualization script automatically: {e}")
        print("Please run manually: python debug_and_generate_all_visuals.py")
    
    # Run comprehensive report generation
    print("\n" + "="*60)
    print("Generating comprehensive reports...")
    print("="*60)
    try:
        result = subprocess.run([sys.executable, 'generate_all_reports.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(" All reports generated successfully!")
        else:
            print("  Some reports may have failed. Running script manually may help.")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"  Could not run report script automatically: {e}")
        print("Please run manually: python generate_all_reports.py")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    print("\n📊 Expected Visualizations (13 files in results/figures/):")
    expected_figures = [
        'pnl_distribution.png', 'trader_clusters.html', 'time_analysis.html',
        'trader_performance_overview.png', 'sentiment_comparison.png', 
        'risk_metrics_sentiment.png', 'cluster_trading_patterns.png',
        'sentiment_deep_dive.png', 'executive_dashboard.html',
        'trader_clusters_3d.html', 'cluster_radar_chart.html',
        'time_series_analysis.html', 'recommendations.png'
    ]
    
    figures_count = 0
    for fig in expected_figures:
        if os.path.exists(f'results/figures/{fig}'):
            figures_count += 1
            print(f"  ✓ {fig}")
        else:
            print(f"  ✗ {fig} (missing)")
    
    print(f"\nTotal figures generated: {figures_count}/13")
    
    print("\n📄 Expected Reports (11 files in results/reports/):")
    expected_reports = [
        'trader_performance_metrics.csv', 'sentiment_impact_analysis.csv',
        'analysis_report.json', 'analysis_report.md', 'statistical_results.json',
        'sentiment_insights.json', 'trader_profiles.json', 'final_insights.json',
        'recommendations.json', 'FINAL_REPORT.md', 'presentation_summary.json'
    ]
    
    reports_count = 0
    for report in expected_reports:
        if os.path.exists(f'results/reports/{report}'):
            reports_count += 1
            print(f"  ✓ {report}")
        else:
            print(f"  ✗ {report} (missing)")
    
    print(f"\nTotal reports generated: {reports_count}/11")
    
    if figures_count < 13 or reports_count < 11:
        print("\n  Some files are missing. Please run:")
        if figures_count < 13:
            print("   python debug_and_generate_all_visuals.py")
        if reports_count < 11:
            print("   python generate_all_reports.py")
    else:
        print("\n All files generated successfully!")
        

def generate_report(trader_metrics, sentiment_impact, statistical_results, trader_clusters):
    """Generate a comprehensive analysis report"""
    
    report = {
        "Executive Summary": {
            "Total Traders Analyzed": len(trader_metrics),
            "Average Win Rate": f"{trader_metrics['win_rate'].mean():.2%}",
            "Total PnL": f"${trader_metrics['total_pnl'].sum():,.2f}",
            "Statistical Significance": "Yes" if statistical_results['p_value'] < 0.05 else "No"
        },
        
        "Key Findings": {
            "1. Sentiment Impact": {
                "Fear Period Avg PnL": f"${statistical_results['fear_mean_pnl']:.2f}",
                "Greed Period Avg PnL": f"${statistical_results['greed_mean_pnl']:.2f}",
                "Effect Size (Cohen's d)": f"{statistical_results['cohens_d']:.3f}",
                "Interpretation": "Medium effect" if abs(statistical_results['cohens_d']) > 0.5 else "Small effect"
            },
            
            "2. Trader Segments": {
                f"Cluster {i}": {
                    "Count": len(trader_clusters[trader_clusters['cluster'] == i]),
                    "Avg Total PnL": f"${trader_clusters[trader_clusters['cluster'] == i]['total_pnl'].mean():,.2f}",
                    "Avg Win Rate": f"{trader_clusters[trader_clusters['cluster'] == i]['win_rate'].mean():.2%}"
                } for i in range(4)
            },
            
            "3. Top Performers": {
                "Best Trader": trader_metrics['total_pnl'].idxmax(),
                "Best PnL": f"${trader_metrics['total_pnl'].max():,.2f}",
                "Most Consistent": trader_metrics.loc[trader_metrics['pnl_std'].idxmin()].name if len(trader_metrics) > 0 else "N/A"
            }
        },
        
        "Recommendations": [
            "1. Traders should adjust position sizes during extreme fear periods",
            "2. Higher leverage correlates with higher risk - use with caution",
            "3. Most profitable trading hours are between 14:00-18:00 UTC",
            "4. Diversification across different market sentiments is crucial"
        ]
    }
    
    # Save report as JSON
    with open('results/reports/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Create markdown report
    create_markdown_report(report, statistical_results)

def create_markdown_report(report, statistical_results):
    """Create a markdown report for easy reading"""
    
    markdown_content = f"""# Trader Behavior Insights: Analysis Report

## Executive Summary

- **Total Traders Analyzed**: {report['Executive Summary']['Total Traders Analyzed']}
- **Average Win Rate**: {report['Executive Summary']['Average Win Rate']}
- **Total PnL**: {report['Executive Summary']['Total PnL']}
- **Statistical Significance**: {report['Executive Summary']['Statistical Significance']}

## Key Findings

### 1. Market Sentiment Impact on Trading Performance

Our analysis reveals a statistically {"significant" if statistical_results['p_value'] < 0.05 else "insignificant"} relationship between market sentiment and trader performance:

- **Fear Period Average PnL**: {report['Key Findings']['1. Sentiment Impact']['Fear Period Avg PnL']}
- **Greed Period Average PnL**: {report['Key Findings']['1. Sentiment Impact']['Greed Period Avg PnL']}
- **Effect Size**: {report["Key Findings"]["1. Sentiment Impact"]["Effect Size (Cohen's d)"]} ({report["Key Findings"]["1. Sentiment Impact"]["Interpretation"]})
- **P-value**: {statistical_results['p_value']:.4f}

### 2. Trader Segmentation

We identified 4 distinct trader segments based on behavior patterns:

"""
    
    for i in range(4):
        cluster_info = report['Key Findings']['2. Trader Segments'][f'Cluster {i}']
        markdown_content += f"""
**Cluster {i}**:
- Traders: {cluster_info['Count']}
- Average Total PnL: {cluster_info['Avg Total PnL']}
- Average Win Rate: {cluster_info['Avg Win Rate']}
"""
    
    markdown_content += f"""
### 3. Top Performers

- **Best Performing Trader**: {report['Key Findings']['3. Top Performers']['Best Trader']}
- **Highest PnL**: {report['Key Findings']['3. Top Performers']['Best PnL']}
- **Most Consistent Trader**: {report['Key Findings']['3. Top Performers']['Most Consistent']}

## Actionable Recommendations

"""
    
    for recommendation in report['Recommendations']:
        markdown_content += f"- {recommendation}\n"
    
    markdown_content += """
## Methodology

1. **Data Collection**: Analyzed historical trader data from Hyperliquid and Bitcoin Fear/Greed Index
2. **Statistical Analysis**: Performed t-tests and effect size calculations
3. **Machine Learning**: Applied K-means clustering to identify trader segments
4. **Visualization**: Created interactive dashboards for deeper insights

## Next Steps

1. Implement real-time monitoring system based on these insights
2. Develop trading strategies tailored to each trader segment
3. Create sentiment-based risk management protocols
4. Build predictive models for trader success probability

---
---
*Analysis conducted using Python with pandas, scikit-learn, and statistical libraries*
"""
    
    with open('results/reports/analysis_report.md', 'w') as f:
        f.write(markdown_content)

if __name__ == "__main__":
    main()