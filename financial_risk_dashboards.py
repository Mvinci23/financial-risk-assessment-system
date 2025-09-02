"""
Financial Risk Assessment & Fraud Detection System
Dashboard Creation Module - Generate comprehensive visualizations and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Financial theme colors
FINANCIAL_COLORS = {
    'primary': '#1f4e79',      # Dark blue
    'secondary': '#dc3545',    # Red (for risk/fraud)
    'success': '#28a745',      # Green (for good performance)
    'warning': '#ffc107',      # Yellow (for caution)
    'info': '#17a2b8',         # Teal (for information)
    'dark': '#343a40',         # Dark gray
    'light': '#f8f9fa'         # Light gray
}

print("=" * 80)
print("🏦 FINANCIAL RISK ASSESSMENT & FRAUD DETECTION SYSTEM")
print("=" * 80)
print("📊 COMPREHENSIVE DASHBOARD CREATION MODULE")
print("-" * 40)

class FinancialDashboardCreator:
    """Create comprehensive financial risk assessment dashboards"""
    
    def __init__(self):
        self.data = {}
        self.colors = FINANCIAL_COLORS
        
    def load_all_data(self):
        """Load all financial datasets and analysis results"""
        print("📂 Loading datasets and analysis results...")
        
        # Load datasets
        datasets = {
            'transactions': 'credit_card_transactions.csv',
            'customers': 'customer_profiles.csv',
            'loans': 'loan_applications.csv',
            'accounts': 'bank_accounts.csv',
            'indicators': 'economic_indicators.csv'
        }
        
        for name, filename in datasets.items():
            try:
                self.data[name] = pd.read_csv(f'data/{filename}')
                print(f"✅ Loaded {name}: {self.data[name].shape}")
            except FileNotFoundError:
                print(f"⚠️ Dataset {filename} not found - some dashboards may be limited")
        
        # Load analysis results if available
        try:
            with open('reports/financial_risk_analysis.json', 'r') as f:
                self.analysis_results = json.load(f)
            print("✅ Loaded analysis results")
        except FileNotFoundError:
            print("⚠️ Analysis results not found - run financial_risk_analysis.py first")
            self.analysis_results = {}
        
        # Load model results if available
        try:
            with open('models/model_results.json', 'r') as f:
                self.model_results = json.load(f)
            print("✅ Loaded model results")
        except FileNotFoundError:
            print("⚠️ Model results not found - run financial_risk_models.py first")
            self.model_results = {}
        
        # Convert date columns
        if 'transactions' in self.data:
            self.data['transactions']['transaction_date'] = pd.to_datetime(self.data['transactions']['transaction_date'])
        if 'loans' in self.data:
            self.data['loans']['application_date'] = pd.to_datetime(self.data['loans']['application_date'])
        if 'accounts' in self.data:
            self.data['accounts']['account_open_date'] = pd.to_datetime(self.data['accounts']['account_open_date'])
        if 'indicators' in self.data:
            self.data['indicators']['date'] = pd.to_datetime(self.data['indicators']['date'])
        
        print("✅ All available data loaded successfully")
        return True
    
    def create_dashboard_1_fraud_overview(self):
        """Dashboard 1: Fraud Detection Overview"""
        print("📊 Creating Dashboard 1: Fraud Detection Overview...")
        
        if 'transactions' not in self.data:
            print("⚠️ Transaction data not available")
            return
        
        transactions = self.data['transactions']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fraud Detection Overview Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Fraud Rate by Transaction Amount Ranges
        amount_bins = [0, 100, 500, 1000, 5000, float('inf')]
        amount_labels = ['$0-$100', '$100-$500', '$500-$1K', '$1K-$5K', '$5K+']
        transactions['amount_range'] = pd.cut(transactions['amount'], bins=amount_bins, labels=amount_labels)
        
        fraud_by_amount = transactions.groupby('amount_range')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_amount['fraud_rate'] = fraud_by_amount['mean']
        
        bars1 = ax1.bar(range(len(fraud_by_amount)), fraud_by_amount['fraud_rate'], 
                       color=self.colors['secondary'], alpha=0.7)
        ax1.set_xticks(range(len(fraud_by_amount)))
        ax1.set_xticklabels(fraud_by_amount['amount_range'], rotation=45)
        ax1.set_ylabel('Fraud Rate')
        ax1.set_title('Fraud Rate by Transaction Amount', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, fraud_by_amount['fraud_rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Fraud by Category
        category_fraud = transactions.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        category_fraud = category_fraud.sort_values('mean', ascending=True).tail(10)
        
        bars2 = ax2.barh(range(len(category_fraud)), category_fraud['mean'], 
                        color=self.colors['secondary'], alpha=0.7)
        ax2.set_yticks(range(len(category_fraud)))
        ax2.set_yticklabels(category_fraud['category'])
        ax2.set_xlabel('Fraud Rate')
        ax2.set_title('Fraud Rate by Transaction Category', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Fraud Trends Over Time
        daily_fraud = transactions.groupby(transactions['transaction_date'].dt.date)['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        daily_fraud['fraud_rate'] = daily_fraud['mean']
        
        ax3.plot(daily_fraud['transaction_date'], daily_fraud['fraud_rate'], 
                color=self.colors['secondary'], linewidth=2, marker='o', markersize=3)
        ax3.fill_between(daily_fraud['transaction_date'], daily_fraud['fraud_rate'], 
                        alpha=0.3, color=self.colors['secondary'])
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Daily Fraud Rate')
        ax3.set_title('Fraud Rate Trends Over Time', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fraud by Location (Domestic vs International)
        location_fraud = transactions.groupby(transactions['country'] == 'USA')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        location_fraud['location'] = location_fraud['country'].map({True: 'Domestic', False: 'International'})
        
        bars4 = ax4.bar(location_fraud['location'], location_fraud['mean'], 
                       color=[self.colors['success'], self.colors['secondary']], alpha=0.7)
        ax4.set_ylabel('Fraud Rate')
        ax4.set_title('Fraud Rate: Domestic vs International', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars4, location_fraud['mean']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dashboards/01_fraud_detection_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 01_fraud_detection_overview.png")
    
    def create_dashboard_2_credit_risk_analysis(self):
        """Dashboard 2: Credit Risk Analysis"""
        print("📊 Creating Dashboard 2: Credit Risk Analysis...")
        
        if 'loans' not in self.data:
            print("⚠️ Loan data not available")
            return
        
        loans = self.data['loans']
        approved_loans = loans[loans['is_approved'] == 1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Credit Risk Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Default Rate by Credit Score Ranges
        score_bins = [300, 580, 669, 739, 799, 850]
        score_labels = ['Poor\n(300-580)', 'Fair\n(580-669)', 'Good\n(670-739)', 'Very Good\n(740-799)', 'Excellent\n(800-850)']
        approved_loans['score_range'] = pd.cut(approved_loans['credit_score'], bins=score_bins, labels=score_labels)
        
        score_default = approved_loans.groupby('score_range')['is_default'].agg(['count', 'sum', 'mean']).reset_index()
        
        bars1 = ax1.bar(range(len(score_default)), score_default['mean'], 
                       color=self.colors['warning'], alpha=0.7)
        ax1.set_xticks(range(len(score_default)))
        ax1.set_xticklabels(score_default['score_range'])
        ax1.set_ylabel('Default Rate')
        ax1.set_title('Default Rate by Credit Score Range', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars1, score_default['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Loan Performance by Type
        loan_type_default = approved_loans.groupby('loan_type')['is_default'].agg(['count', 'sum', 'mean']).reset_index()
        loan_type_default = loan_type_default.sort_values('mean', ascending=True)
        
        bars2 = ax2.barh(range(len(loan_type_default)), loan_type_default['mean'], 
                        color=self.colors['primary'], alpha=0.7)
        ax2.set_yticks(range(len(loan_type_default)))
        ax2.set_yticklabels(loan_type_default['loan_type'])
        ax2.set_xlabel('Default Rate')
        ax2.set_title('Default Rate by Loan Type', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Loan Amount vs Default Rate
        amount_bins = np.percentile(approved_loans['loan_amount'], [0, 25, 50, 75, 100])
        approved_loans['amount_quartile'] = pd.cut(approved_loans['loan_amount'], bins=amount_bins, 
                                                  labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        amount_default = approved_loans.groupby('amount_quartile')['is_default'].agg(['count', 'mean']).reset_index()
        
        bars3 = ax3.bar(range(len(amount_default)), amount_default['mean'], 
                       color=self.colors['info'], alpha=0.7)
        ax3.set_xticks(range(len(amount_default)))
        ax3.set_xticklabels(amount_default['amount_quartile'])
        ax3.set_ylabel('Default Rate')
        ax3.set_title('Default Rate by Loan Amount Quartile', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. DTI Ratio vs Default Rate
        dti_bins = [0, 0.2, 0.3, 0.4, 1.0]
        dti_labels = ['<20%', '20-30%', '30-40%', '40%+']
        approved_loans['dti_range'] = pd.cut(approved_loans['debt_to_income_ratio'], bins=dti_bins, labels=dti_labels)
        
        dti_default = approved_loans.groupby('dti_range')['is_default'].agg(['count', 'mean']).reset_index()
        
        bars4 = ax4.bar(range(len(dti_default)), dti_default['mean'], 
                       color=self.colors['secondary'], alpha=0.7)
        ax4.set_xticks(range(len(dti_default)))
        ax4.set_xticklabels(dti_default['dti_range'])
        ax4.set_ylabel('Default Rate')
        ax4.set_title('Default Rate by Debt-to-Income Ratio', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/02_credit_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 02_credit_risk_analysis.png")
    
    def create_dashboard_3_account_monitoring(self):
        """Dashboard 3: Account Risk Monitoring"""
        print("📊 Creating Dashboard 3: Account Risk Monitoring...")
        
        if 'accounts' not in self.data:
            print("⚠️ Account data not available")
            return
        
        accounts = self.data['accounts']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Account Risk Monitoring Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Suspicious Activity by Account Type
        account_risk = accounts.groupby('account_type')['suspicious_activity'].agg(['count', 'sum', 'mean']).reset_index()
        account_risk = account_risk.sort_values('mean', ascending=True)
        
        bars1 = ax1.barh(range(len(account_risk)), account_risk['mean'], 
                        color=self.colors['secondary'], alpha=0.7)
        ax1.set_yticks(range(len(account_risk)))
        ax1.set_yticklabels(account_risk['account_type'])
        ax1.set_xlabel('Suspicious Activity Rate')
        ax1.set_title('Suspicious Activity by Account Type', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Balance Distribution
        ax2.hist(accounts['current_balance'], bins=50, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax2.axvline(accounts['current_balance'].mean(), color=self.colors['secondary'], 
                   linestyle='--', linewidth=2, label=f'Mean: ${accounts["current_balance"].mean():,.0f}')
        ax2.axvline(accounts['current_balance'].median(), color=self.colors['success'], 
                   linestyle='--', linewidth=2, label=f'Median: ${accounts["current_balance"].median():,.0f}')
        ax2.set_xlabel('Account Balance ($)')
        ax2.set_ylabel('Number of Accounts')
        ax2.set_title('Account Balance Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction Volume vs Suspicious Activity
        ax3.scatter(accounts['monthly_transactions'], accounts['monthly_volume'], 
                   c=accounts['suspicious_activity'], cmap='RdYlBu_r', alpha=0.6, s=30)
        ax3.set_xlabel('Monthly Transactions')
        ax3.set_ylabel('Monthly Transaction Volume ($)')
        ax3.set_title('Transaction Patterns (Red = Suspicious)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Suspicious Activity')
        
        # 4. Account Age vs Risk
        accounts['account_age_years'] = (datetime.now() - accounts['account_open_date']).dt.days / 365
        age_bins = [0, 1, 3, 5, 10, 100]
        age_labels = ['<1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years']
        accounts['age_range'] = pd.cut(accounts['account_age_years'], bins=age_bins, labels=age_labels)
        
        age_risk = accounts.groupby('age_range')['suspicious_activity'].agg(['count', 'mean']).reset_index()
        
        bars4 = ax4.bar(range(len(age_risk)), age_risk['mean'], 
                       color=self.colors['warning'], alpha=0.7)
        ax4.set_xticks(range(len(age_risk)))
        ax4.set_xticklabels(age_risk['age_range'], rotation=45)
        ax4.set_ylabel('Suspicious Activity Rate')
        ax4.set_title('Suspicious Activity by Account Age', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/03_account_monitoring.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 03_account_monitoring.png")
    
    def create_dashboard_4_economic_indicators(self):
        """Dashboard 4: Economic Indicators Impact"""
        print("📊 Creating Dashboard 4: Economic Indicators Impact...")
        
        if 'indicators' not in self.data:
            print("⚠️ Economic indicators data not available")
            return
        
        indicators = self.data['indicators']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Economic Indicators Impact Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Economic Trends Over Time
        ax1.plot(indicators['date'], indicators['unemployment_rate'], label='Unemployment %', 
                color=self.colors['secondary'], linewidth=2)
        ax1.plot(indicators['date'], indicators['gdp_growth_rate'], label='GDP Growth %', 
                color=self.colors['success'], linewidth=2)
        ax1.plot(indicators['date'], indicators['inflation_rate'], label='Inflation %', 
                color=self.colors['warning'], linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Key Economic Indicators Trends', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Interest Rates
        ax2.plot(indicators['date'], indicators['federal_funds_rate'], label='Federal Funds Rate', 
                color=self.colors['primary'], linewidth=2)
        ax2.plot(indicators['date'], indicators['mortgage_rate'], label='Mortgage Rate', 
                color=self.colors['info'], linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Interest Rate (%)')
        ax2.set_title('Interest Rate Trends', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Market Volatility
        ax3.plot(indicators['date'], indicators['vix_volatility'], color=self.colors['secondary'], linewidth=2)
        ax3.fill_between(indicators['date'], indicators['vix_volatility'], alpha=0.3, color=self.colors['secondary'])
        ax3.axhline(y=30, color=self.colors['warning'], linestyle='--', label='High Volatility (30)')
        ax3.axhline(y=20, color=self.colors['info'], linestyle='--', label='Moderate Volatility (20)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('VIX Volatility Index')
        ax3.set_title('Market Volatility (VIX)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Stock Market Performance
        ax4.plot(indicators['date'], indicators['stock_market_index'], color=self.colors['success'], linewidth=2)
        ax4.fill_between(indicators['date'], indicators['stock_market_index'], alpha=0.3, color=self.colors['success'])
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Stock Market Index')
        ax4.set_title('Stock Market Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('dashboards/04_economic_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 04_economic_indicators.png")
    
    def create_dashboard_5_model_performance(self):
        """Dashboard 5: ML Model Performance"""
        print("📊 Creating Dashboard 5: ML Model Performance...")
        
        if not self.model_results:
            print("⚠️ Model results not available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ML Model Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Fraud Detection Model Comparison
        if 'fraud_detection' in self.model_results.get('model_results', {}):
            fraud_results = self.model_results['model_results']['fraud_detection']['results']
            models = list(fraud_results.keys())
            auc_scores = [fraud_results[model]['auc'] for model in models]
            
            bars1 = ax1.bar(range(len(models)), auc_scores, color=self.colors['primary'], alpha=0.7)
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.set_ylabel('AUC Score')
            ax1.set_title('Fraud Detection Model Performance', fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars1, auc_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Credit Risk Model Comparison
        if 'credit_risk' in self.model_results.get('model_results', {}):
            credit_results = self.model_results['model_results']['credit_risk']['results']
            models = list(credit_results.keys())
            auc_scores = [credit_results[model]['auc'] for model in models]
            
            bars2 = ax2.bar(range(len(models)), auc_scores, color=self.colors['success'], alpha=0.7)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.set_ylabel('AUC Score')
            ax2.set_title('Credit Risk Model Performance', fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars2, auc_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Importance (Fraud Detection)
        if 'feature_importance' in self.model_results.get('model_results', {}):
            if 'fraud_detection' in self.model_results['model_results']['feature_importance']:
                fraud_features = self.model_results['model_results']['feature_importance']['fraud_detection'][:8]
                feature_names = [f[0] for f in fraud_features]
                feature_scores = [f[1] for f in fraud_features]
                
                bars3 = ax3.barh(range(len(feature_names)), feature_scores, 
                               color=self.colors['secondary'], alpha=0.7)
                ax3.set_yticks(range(len(feature_names)))
                ax3.set_yticklabels(feature_names)
                ax3.set_xlabel('Feature Importance')
                ax3.set_title('Top Fraud Detection Features', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        # 4. Model Metrics Comparison
        if 'fraud_detection' in self.model_results.get('model_results', {}):
            fraud_results = self.model_results['model_results']['fraud_detection']['results']
            best_model = self.model_results['model_results']['fraud_detection']['best_model']
            best_results = fraud_results[best_model]
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            values = [best_results[metric] for metric in metrics]
            
            bars4 = ax4.bar(range(len(metrics)), values, color=self.colors['info'], alpha=0.7)
            ax4.set_xticks(range(len(metrics)))
            ax4.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax4.set_ylabel('Score')
            ax4.set_title(f'Best Model Metrics: {best_model}', fontweight='bold')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars4, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dashboards/05_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 05_model_performance.png")
    
    def create_dashboard_6_transaction_patterns(self):
        """Dashboard 6: Transaction Pattern Analysis"""
        print("📊 Creating Dashboard 6: Transaction Pattern Analysis...")
        
        if 'transactions' not in self.data:
            print("⚠️ Transaction data not available")
            return
        
        transactions = self.data['transactions']
        transactions['transaction_time'] = pd.to_datetime(transactions['transaction_time'], format='%H:%M:%S')
        transactions['hour'] = transactions['transaction_time'].dt.hour
        transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Transaction Pattern Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Hourly Transaction Patterns
        hourly_patterns = transactions.groupby('hour').agg({
            'transaction_id': 'count',
            'is_fraud': ['sum', 'mean']
        }).reset_index()
        hourly_patterns.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']
        
        ax1_twin = ax1.twinx()
        bars1 = ax1.bar(hourly_patterns['hour'], hourly_patterns['total_transactions'], 
                       color=self.colors['primary'], alpha=0.7, label='Total Transactions')
        line1 = ax1_twin.plot(hourly_patterns['hour'], hourly_patterns['fraud_rate'], 
                             color=self.colors['secondary'], linewidth=3, marker='o', label='Fraud Rate')
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Transactions', color=self.colors['primary'])
        ax1_twin.set_ylabel('Fraud Rate', color=self.colors['secondary'])
        ax1.set_title('Transaction Volume and Fraud by Hour', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Day of Week Patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_patterns = transactions.groupby('day_of_week').agg({
            'transaction_id': 'count',
            'amount': 'mean',
            'is_fraud': 'mean'
        }).reset_index()
        daily_patterns['day_of_week'] = pd.Categorical(daily_patterns['day_of_week'], categories=day_order, ordered=True)
        daily_patterns = daily_patterns.sort_values('day_of_week')
        
        ax2_twin = ax2.twinx()
        bars2 = ax2.bar(range(len(daily_patterns)), daily_patterns['transaction_id'], 
                       color=self.colors['success'], alpha=0.7)
        line2 = ax2_twin.plot(range(len(daily_patterns)), daily_patterns['is_fraud'], 
                             color=self.colors['secondary'], linewidth=3, marker='o')
        
        ax2.set_xticks(range(len(daily_patterns)))
        ax2.set_xticklabels(daily_patterns['day_of_week'], rotation=45)
        ax2.set_ylabel('Number of Transactions', color=self.colors['success'])
        ax2_twin.set_ylabel('Fraud Rate', color=self.colors['secondary'])
        ax2.set_title('Transaction Volume and Fraud by Day', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Amount Distribution by Category
        category_amounts = transactions.groupby('category')['amount'].agg(['mean', 'median', 'std']).reset_index()
        category_amounts = category_amounts.sort_values('mean', ascending=True).tail(8)
        
        bars3 = ax3.barh(range(len(category_amounts)), category_amounts['mean'], 
                        color=self.colors['info'], alpha=0.7)
        ax3.set_yticks(range(len(category_amounts)))
        ax3.set_yticklabels(category_amounts['category'])
        ax3.set_xlabel('Average Transaction Amount ($)')
        ax3.set_title('Average Transaction Amount by Category', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Geographic Risk Distribution
        geographic_risk = transactions.groupby('state').agg({
            'transaction_id': 'count',
            'is_fraud': ['sum', 'mean'],
            'amount': 'mean'
        }).reset_index()
        geographic_risk.columns = ['state', 'total_transactions', 'fraud_count', 'fraud_rate', 'avg_amount']
        geographic_risk = geographic_risk[geographic_risk['total_transactions'] >= 100].sort_values('fraud_rate', ascending=True)
        
        bars4 = ax4.barh(range(len(geographic_risk)), geographic_risk['fraud_rate'], 
                        color=self.colors['warning'], alpha=0.7)
        ax4.set_yticks(range(len(geographic_risk)))
        ax4.set_yticklabels(geographic_risk['state'])
        ax4.set_xlabel('Fraud Rate')
        ax4.set_title('Fraud Rate by State', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/06_transaction_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 06_transaction_patterns.png")
    
    def create_dashboard_7_customer_segmentation(self):
        """Dashboard 7: Customer Risk Segmentation"""
        print("📊 Creating Dashboard 7: Customer Risk Segmentation...")
        
        if 'customers' not in self.data:
            print("⚠️ Customer data not available")
            return
        
        customers = self.data['customers']
        
        # Merge with transaction data for additional insights
        if 'transactions' in self.data:
            customer_stats = self.data['transactions'].groupby('customer_id').agg({
                'amount': ['sum', 'mean', 'count'],
                'is_fraud': 'sum'
            }).reset_index()
            customer_stats.columns = ['customer_id', 'total_spent', 'avg_transaction', 'transaction_count', 'fraud_incidents']
            customers = customers.merge(customer_stats, on='customer_id', how='left')
            customers = customers.fillna(0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Risk Segmentation Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Risk Profile Distribution
        risk_dist = customers['risk_profile'].value_counts()
        
        colors_risk = [self.colors['success'], self.colors['warning'], self.colors['secondary']]
        wedges, texts, autotexts = ax1.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%', 
                                          colors=colors_risk, startangle=90)
        ax1.set_title('Customer Risk Profile Distribution', fontweight='bold')
        
        # 2. Age vs Credit Score
        ax2.scatter(customers['age'], customers['credit_score'], 
                   c=customers['risk_profile'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}), 
                   cmap='RdYlGn_r', alpha=0.6, s=30)
        ax2.set_xlabel('Customer Age')
        ax2.set_ylabel('Credit Score')
        ax2.set_title('Age vs Credit Score (Color = Risk Level)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Income Distribution by Risk Profile
        risk_profiles = ['LOW', 'MEDIUM', 'HIGH']
        colors_income = [self.colors['success'], self.colors['warning'], self.colors['secondary']]
        
        for i, risk in enumerate(risk_profiles):
            risk_data = customers[customers['risk_profile'] == risk]['income']
            ax3.hist(risk_data, bins=30, alpha=0.6, label=f'{risk} Risk', 
                    color=colors_income[i], density=True)
        
        ax3.set_xlabel('Annual Income ($)')
        ax3.set_ylabel('Density')
        ax3.set_title('Income Distribution by Risk Profile', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Account Age vs Fraud Incidents
        if 'transaction_count' in customers.columns:
            customers_with_activity = customers[customers['transaction_count'] > 0]
            
            ax4.scatter(customers_with_activity['account_age_months'], 
                       customers_with_activity['fraud_incidents'], 
                       alpha=0.6, s=30, color=self.colors['secondary'])
            
            ax4.set_xlabel('Account Age (Months)')
            ax4.set_ylabel('Fraud Incidents')
            ax4.set_title('Account Age vs Fraud Incidents', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Transaction data not available\nfor fraud incident analysis', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Account Age vs Fraud Incidents', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dashboards/07_customer_segmentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 07_customer_segmentation.png")
    
    def create_dashboard_8_portfolio_analysis(self):
        """Dashboard 8: Loan Portfolio Analysis"""
        print("📊 Creating Dashboard 8: Loan Portfolio Analysis...")
        
        if 'loans' not in self.data:
            print("⚠️ Loan data not available")
            return
        
        loans = self.data['loans']
        approved_loans = loans[loans['is_approved'] == 1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Loan Portfolio Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Loan Portfolio Composition
        portfolio_by_type = approved_loans.groupby('loan_type')['loan_amount'].sum().reset_index()
        portfolio_by_type = portfolio_by_type.sort_values('loan_amount', ascending=False)
        
        wedges, texts, autotexts = ax1.pie(portfolio_by_type['loan_amount'], labels=portfolio_by_type['loan_type'], 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Loan Portfolio by Type ($ Value)', fontweight='bold')
        
        # 2. Interest Rate Distribution
        ax2.hist(approved_loans['interest_rate'] * 100, bins=30, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax2.axvline(approved_loans['interest_rate'].mean() * 100, color=self.colors['secondary'], 
                   linestyle='--', linewidth=2, label=f'Mean: {approved_loans["interest_rate"].mean()*100:.2f}%')
        ax2.set_xlabel('Interest Rate (%)')
        ax2.set_ylabel('Number of Loans')
        ax2.set_title('Interest Rate Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Loan Performance Trends
        approved_loans['application_month'] = approved_loans['application_date'].dt.to_period('M')
        monthly_performance = approved_loans.groupby('application_month').agg({
            'loan_amount': 'sum',
            'is_default': 'mean'
        }).reset_index()
        
        ax3_twin = ax3.twinx()
        bars3 = ax3.bar(range(len(monthly_performance)), monthly_performance['loan_amount'] / 1e6, 
                       color=self.colors['success'], alpha=0.7)
        line3 = ax3_twin.plot(range(len(monthly_performance)), monthly_performance['is_default'], 
                             color=self.colors['secondary'], linewidth=3, marker='o')
        
        ax3.set_xticks(range(0, len(monthly_performance), 3))
        ax3.set_xticklabels([str(monthly_performance.iloc[i]['application_month']) 
                           for i in range(0, len(monthly_performance), 3)], rotation=45)
        ax3.set_ylabel('Loan Volume ($M)', color=self.colors['success'])
        ax3_twin.set_ylabel('Default Rate', color=self.colors['secondary'])
        ax3.set_title('Monthly Loan Volume vs Default Rate', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return Analysis
        risk_return = approved_loans.groupby('loan_type').agg({
            'interest_rate': 'mean',
            'is_default': 'mean',
            'loan_amount': 'sum'
        }).reset_index()
        
        # Create bubble chart
        ax4.scatter(risk_return['is_default'], risk_return['interest_rate'] * 100, 
                   s=risk_return['loan_amount'] / 1e6, alpha=0.6, c=range(len(risk_return)), cmap='viridis')
        
        # Add labels
        for i, row in risk_return.iterrows():
            ax4.annotate(row['loan_type'], (row['is_default'], row['interest_rate'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Default Rate')
        ax4.set_ylabel('Average Interest Rate (%)')
        ax4.set_title('Risk-Return Analysis by Loan Type\n(Bubble Size = Portfolio Value)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/08_portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 08_portfolio_analysis.png")
    
    def create_dashboard_9_risk_correlation(self):
        """Dashboard 9: Risk Factor Correlation Analysis"""
        print("📊 Creating Dashboard 9: Risk Factor Correlation Analysis...")
        
        # Combine relevant data for correlation analysis
        correlation_data = pd.DataFrame()
        
        if 'transactions' in self.data and len(self.data['transactions']) > 0:
            trans_corr = self.data['transactions'][['amount', 'customer_age', 'customer_income', 
                                                   'customer_credit_score', 'is_fraud']].copy()
            trans_corr['data_source'] = 'transactions'
            correlation_data = pd.concat([correlation_data, trans_corr])
        
        if 'loans' in self.data and len(self.data['loans']) > 0:
            loan_corr = self.data['loans'][['loan_amount', 'applicant_age', 'annual_income', 
                                          'credit_score', 'debt_to_income_ratio', 'interest_rate', 'is_default']].copy()
            loan_corr.rename(columns={'loan_amount': 'amount', 'applicant_age': 'customer_age', 
                                    'annual_income': 'customer_income', 'credit_score': 'customer_credit_score'}, inplace=True)
            loan_corr['is_fraud'] = loan_corr['is_default']  # For correlation purposes
            correlation_data = pd.concat([correlation_data, loan_corr[['amount', 'customer_age', 'customer_income', 
                                                                      'customer_credit_score', 'is_fraud']]])
        
        if correlation_data.empty:
            print("⚠️ No correlation data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Factor Correlation Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Correlation Heatmap
        corr_matrix = correlation_data.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Risk Factor Correlation Matrix', fontweight='bold')
        
        # 2. Credit Score vs Risk
        ax2.scatter(correlation_data['customer_credit_score'], correlation_data['is_fraud'], 
                   alpha=0.5, s=20, color=self.colors['secondary'])
        
        # Add trend line
        z = np.polyfit(correlation_data['customer_credit_score'].dropna(), 
                      correlation_data['is_fraud'].dropna(), 1)
        p = np.poly1d(z)
        ax2.plot(correlation_data['customer_credit_score'], p(correlation_data['customer_credit_score']), 
                "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Credit Score')
        ax2.set_ylabel('Risk Indicator')
        ax2.set_title('Credit Score vs Risk Relationship', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Income vs Amount Risk
        ax3.scatter(correlation_data['customer_income'], correlation_data['amount'], 
                   c=correlation_data['is_fraud'], cmap='RdYlBu_r', alpha=0.6, s=20)
        ax3.set_xlabel('Customer Income ($)')
        ax3.set_ylabel('Transaction/Loan Amount ($)')
        ax3.set_title('Income vs Amount (Color = Risk)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Age-based Risk Distribution
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        correlation_data['age_group'] = pd.cut(correlation_data['customer_age'], bins=age_bins, labels=age_labels)
        
        age_risk = correlation_data.groupby('age_group')['is_fraud'].agg(['count', 'mean']).reset_index()
        age_risk = age_risk.dropna()
        
        bars4 = ax4.bar(range(len(age_risk)), age_risk['mean'], color=self.colors['warning'], alpha=0.7)
        ax4.set_xticks(range(len(age_risk)))
        ax4.set_xticklabels(age_risk['age_group'])
        ax4.set_ylabel('Risk Rate')
        ax4.set_title('Risk Distribution by Age Group', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars4, age_risk['mean']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dashboards/09_risk_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 09_risk_correlation.png")
    
    def create_dashboard_10_executive_summary(self):
        """Dashboard 10: Executive Summary"""
        print("📊 Creating Dashboard 10: Executive Summary...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Financial Risk Assessment - Executive Summary Dashboard', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Create custom layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Key Metrics Cards (Top Row)
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        
        # Calculate key metrics
        metrics_text = "KEY PERFORMANCE INDICATORS\n\n"
        
        if 'transactions' in self.data:
            total_transactions = len(self.data['transactions'])
            fraud_rate = self.data['transactions']['is_fraud'].mean()
            avg_transaction = self.data['transactions']['amount'].mean()
            metrics_text += f"TRANSACTIONS: {total_transactions:,}    FRAUD RATE: {fraud_rate:.3%}    AVG AMOUNT: ${avg_transaction:,.0f}\n"
        
        if 'loans' in self.data:
            total_loans = len(self.data['loans'])
            approval_rate = self.data['loans']['is_approved'].mean()
            default_rate = self.data['loans'][self.data['loans']['is_approved'] == 1]['is_default'].mean()
            metrics_text += f"LOAN APPLICATIONS: {total_loans:,}    APPROVAL RATE: {approval_rate:.1%}    DEFAULT RATE: {default_rate:.1%}\n"
        
        if 'accounts' in self.data:
            total_accounts = len(self.data['accounts'])
            suspicious_rate = self.data['accounts']['suspicious_activity'].mean()
            avg_balance = self.data['accounts']['current_balance'].mean()
            metrics_text += f"BANK ACCOUNTS: {total_accounts:,}    SUSPICIOUS ACTIVITY: {suspicious_rate:.1%}    AVG BALANCE: ${avg_balance:,.0f}"
        
        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Main visualizations
        ax1 = fig.add_subplot(gs[1, :2])  # Risk overview
        ax2 = fig.add_subplot(gs[1, 2:])  # Model performance
        ax3 = fig.add_subplot(gs[2, :2])  # Portfolio composition
        ax4 = fig.add_subplot(gs[2, 2:])  # Economic indicators
        ax5 = fig.add_subplot(gs[3, :])   # Risk trends
        
        # 1. Overall Risk Assessment
        risk_categories = ['Fraud Risk', 'Credit Risk', 'Operational Risk', 'Market Risk']
        
        # Calculate risk levels based on available data
        risk_levels = []
        if 'transactions' in self.data:
            fraud_risk = min(self.data['transactions']['is_fraud'].mean() * 100, 5)  # Cap at 5
            risk_levels.append(fraud_risk)
        else:
            risk_levels.append(2)
        
        if 'loans' in self.data:
            credit_risk = min(self.data['loans'][self.data['loans']['is_approved'] == 1]['is_default'].mean() * 10, 5)
            risk_levels.append(credit_risk)
        else:
            risk_levels.append(3)
        
        if 'accounts' in self.data:
            operational_risk = min(self.data['accounts']['suspicious_activity'].mean() * 20, 5)
            risk_levels.append(operational_risk)
        else:
            risk_levels.append(2.5)
        
        # Market risk (simulated)
        risk_levels.append(3.0)
        
        colors_risk = [self.colors['success'] if r < 2 else self.colors['warning'] if r < 4 else self.colors['secondary'] 
                      for r in risk_levels]
        
        bars1 = ax1.bar(risk_categories, risk_levels, color=colors_risk, alpha=0.7)
        ax1.set_ylabel('Risk Level (1-5 Scale)')
        ax1.set_title('Overall Risk Assessment', fontweight='bold')
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
        
        # Add risk level labels
        for bar, level in zip(bars1, risk_levels):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{level:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Model Performance Summary
        if self.model_results and 'model_results' in self.model_results:
            model_performance = []
            model_names = []
            
            if 'fraud_detection' in self.model_results['model_results']:
                fraud_best = self.model_results['model_results']['fraud_detection']['best_model']
                fraud_auc = self.model_results['model_results']['fraud_detection']['results'][fraud_best]['auc']
                model_names.append('Fraud Detection')
                model_performance.append(fraud_auc)
            
            if 'credit_risk' in self.model_results['model_results']:
                credit_best = self.model_results['model_results']['credit_risk']['best_model']
                credit_auc = self.model_results['model_results']['credit_risk']['results'][credit_best]['auc']
                model_names.append('Credit Risk')
                model_performance.append(credit_auc)
            
            if model_performance:
                bars2 = ax2.bar(model_names, model_performance, color=self.colors['primary'], alpha=0.7)
                ax2.set_ylabel('AUC Score')
                ax2.set_title('ML Model Performance', fontweight='bold')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                for bar, perf in zip(bars2, model_performance):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'Model Performance\nData Not Available', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('ML Model Performance', fontweight='bold')
        
        # 3. Portfolio Composition
        if 'loans' in self.data:
            approved_loans = self.data['loans'][self.data['loans']['is_approved'] == 1]
            portfolio_composition = approved_loans.groupby('loan_type')['loan_amount'].sum()
            
            wedges, texts, autotexts = ax3.pie(portfolio_composition.values, labels=portfolio_composition.index, 
                                              autopct='%1.1f%%', startangle=90)
            ax3.set_title('Loan Portfolio Composition', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Portfolio Data\nNot Available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Loan Portfolio Composition', fontweight='bold')
        
        # 4. Economic Risk Indicators
        if 'indicators' in self.data:
            indicators = self.data['indicators']
            latest = indicators.iloc[-1]
            
            econ_indicators = ['Unemployment\nRate', 'Inflation\nRate', 'Market\nVolatility']
            econ_values = [latest['unemployment_rate'], latest['inflation_rate'], latest['vix_volatility']/10]  # Scale VIX
            
            colors_econ = []
            for val, indicator in zip(econ_values, econ_indicators):
                if 'Unemployment' in indicator and val > 6:
                    colors_econ.append(self.colors['secondary'])
                elif 'Inflation' in indicator and val > 3:
                    colors_econ.append(self.colors['secondary'])
                elif 'Volatility' in indicator and val > 3:
                    colors_econ.append(self.colors['secondary'])
                else:
                    colors_econ.append(self.colors['success'])
            
            bars4 = ax4.bar(econ_indicators, econ_values, color=colors_econ, alpha=0.7)
            ax4.set_ylabel('Percentage / Scaled Value')
            ax4.set_title('Economic Risk Indicators', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            for bar, val in zip(bars4, econ_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Risk Trends Summary
        if 'transactions' in self.data:
            # Monthly fraud trends
            monthly_fraud = self.data['transactions'].groupby(
                self.data['transactions']['transaction_date'].dt.to_period('M')
            )['is_fraud'].mean().reset_index()
            
            ax5.plot(range(len(monthly_fraud)), monthly_fraud['is_fraud'], 
                    color=self.colors['secondary'], linewidth=3, marker='o', markersize=4)
            ax5.fill_between(range(len(monthly_fraud)), monthly_fraud['is_fraud'], 
                           alpha=0.3, color=self.colors['secondary'])
            ax5.set_xlabel('Time Period (Monthly)')
            ax5.set_ylabel('Fraud Rate')
            ax5.set_title('Risk Trends Over Time', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Risk Trend Data Not Available', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Risk Trends Over Time', fontweight='bold')
        
        plt.savefig('dashboards/10_executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 10_executive_summary.png")
    
    def create_all_dashboards(self):
        """Create all financial risk dashboards"""
        print("🚀 Creating All Financial Risk Assessment Dashboards...")
        print("This will generate 10+ comprehensive visualization dashboards")
        print("-" * 60)
        
        # Load all data
        self.load_all_data()
        
        # Create dashboards directory
        os.makedirs('dashboards', exist_ok=True)
        
        # Create all dashboards
        dashboard_functions = [
            self.create_dashboard_1_fraud_overview,
            self.create_dashboard_2_credit_risk_analysis,
            self.create_dashboard_3_account_monitoring,
            self.create_dashboard_4_economic_indicators,
            self.create_dashboard_5_model_performance,
            self.create_dashboard_6_transaction_patterns,
            self.create_dashboard_7_customer_segmentation,
            self.create_dashboard_8_portfolio_analysis,
            self.create_dashboard_9_risk_correlation,
            self.create_dashboard_10_executive_summary
        ]
        
        created_dashboards = []
        for i, dashboard_func in enumerate(dashboard_functions, 1):
            try:
                print(f"\n📊 Creating Dashboard {i}/10...")
                dashboard_func()
                created_dashboards.append(i)
            except Exception as e:
                print(f"⚠️ Error creating dashboard {i}: {e}")
        
        # Summary
        print(f"\n" + "="*80)
        print("🎉 DASHBOARD CREATION COMPLETED!")
        print("="*80)
        print(f"📊 Dashboards Created: {len(created_dashboards)}/10")
        print(f"📁 Location: dashboards/ directory")
        
        print(f"\n📋 Dashboard List:")
        dashboard_names = [
            "01_fraud_detection_overview.png - Fraud patterns and trends",
            "02_credit_risk_analysis.png - Credit risk assessment metrics", 
            "03_account_monitoring.png - Account risk monitoring",
            "04_economic_indicators.png - Economic environment analysis",
            "05_model_performance.png - ML model evaluation results",
            "06_transaction_patterns.png - Transaction behavior analysis",
            "07_customer_segmentation.png - Customer risk profiling",
            "08_portfolio_analysis.png - Loan portfolio composition",
            "09_risk_correlation.png - Risk factor relationships",
            "10_executive_summary.png - High-level executive overview"
        ]
        
        for i, name in enumerate(dashboard_names, 1):
            status = "✅" if i in created_dashboards else "❌"
            print(f"   {status} {name}")
        
        print(f"\n💡 Usage:")
        print(f"   • Use for executive presentations and board meetings")
        print(f"   • Monitor risk levels and model performance")
        print(f"   • Identify trends and patterns for strategic planning")
        print(f"   • Support regulatory reporting and compliance")
        
        print("="*80)
        
        return len(created_dashboards)

def main():
    """Main function to create all dashboards"""
    creator = FinancialDashboardCreator()
    dashboards_created = creator.create_all_dashboards()
    
    if dashboards_created >= 8:
        print("✅ Dashboard creation completed successfully!")
        print(f"🎯 Generated {dashboards_created} professional dashboards")
    else:
        print("⚠️ Some dashboards could not be created - check data availability")
        
    return dashboards_created

if __name__ == "__main__":
    main()