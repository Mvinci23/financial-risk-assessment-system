"""
Financial Risk Assessment & Fraud Detection System
Analysis Module - Comprehensive financial data analysis and insights
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏦 FINANCIAL RISK ASSESSMENT & FRAUD DETECTION SYSTEM")
print("=" * 80)
print("🔍 COMPREHENSIVE ANALYSIS MODULE")
print("-" * 40)

class FinancialRiskAnalyzer:
    """Comprehensive financial risk and fraud analysis"""
    
    def __init__(self):
        self.data = {}
        self.analysis_results = {}
        
    def load_all_datasets(self):
        """Load all financial datasets"""
        print("📂 Loading financial datasets...")
        
        datasets = {
            'transactions': 'credit_card_transactions.csv',
            'customers': 'customer_profiles.csv',
            'loans': 'loan_applications.csv',
            'accounts': 'bank_accounts.csv',
            'indicators': 'economic_indicators.csv'
        }
        
        for name, filename in datasets.items():
            try:
                filepath = f'data/{filename}'
                self.data[name] = pd.read_csv(filepath)
                print(f"✅ Loaded {name}: {self.data[name].shape}")
            except FileNotFoundError:
                print(f"❌ Dataset {filename} not found. Run financial_data_generator.py first.")
                return False
        
        # Convert date columns
        self.data['transactions']['transaction_date'] = pd.to_datetime(self.data['transactions']['transaction_date'])
        self.data['loans']['application_date'] = pd.to_datetime(self.data['loans']['application_date'])
        self.data['accounts']['account_open_date'] = pd.to_datetime(self.data['accounts']['account_open_date'])
        self.data['indicators']['date'] = pd.to_datetime(self.data['indicators']['date'])
        
        print("✅ All datasets loaded successfully")
        return True
    
    def analyze_fraud_patterns(self):
        """Analyze credit card fraud patterns"""
        print("\n🔍 FRAUD PATTERN ANALYSIS")
        print("-" * 50)
        
        transactions = self.data['transactions']
        fraud_analysis = {}
        
        # Basic fraud statistics
        total_transactions = len(transactions)
        fraud_transactions = transactions[transactions['is_fraud'] == 1]
        fraud_count = len(fraud_transactions)
        fraud_rate = fraud_count / total_transactions
        
        print(f"📊 Basic Fraud Statistics:")
        print(f"   Total Transactions: {total_transactions:,}")
        print(f"   Fraudulent Transactions: {fraud_count:,}")
        print(f"   Fraud Rate: {fraud_rate:.3%}")
        
        # Fraud by amount ranges
        print(f"\n💰 Fraud by Transaction Amount:")
        amount_ranges = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, float('inf'))]
        for low, high in amount_ranges:
            if high == float('inf'):
                mask = transactions['amount'] >= low
                range_label = f"${low:,}+"
            else:
                mask = (transactions['amount'] >= low) & (transactions['amount'] < high)
                range_label = f"${low:,}-${high:,}"
            
            range_transactions = transactions[mask]
            range_fraud_rate = range_transactions['is_fraud'].mean() if len(range_transactions) > 0 else 0
            
            print(f"   {range_label}: {range_fraud_rate:.3%} fraud rate ({len(range_transactions):,} transactions)")
        
        # Fraud by category
        print(f"\n🏪 Fraud by Category:")
        category_fraud = transactions.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).round(4)
        category_fraud.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
        category_fraud = category_fraud.sort_values('Fraud_Rate', ascending=False)
        
        for category, row in category_fraud.head(8).iterrows():
            print(f"   {category}: {row['Fraud_Rate']:.3%} ({row['Fraud_Count']:.0f}/{row['Total_Transactions']:.0f})")
        
        # Fraud by location
        print(f"\n🌍 Fraud by Location:")
        # International vs domestic
        domestic_transactions = transactions[transactions['country'] == 'USA']
        international_transactions = transactions[transactions['country'] != 'USA']
        
        domestic_fraud_rate = domestic_transactions['is_fraud'].mean()
        international_fraud_rate = international_transactions['is_fraud'].mean()
        
        print(f"   Domestic (USA): {domestic_fraud_rate:.3%}")
        print(f"   International: {international_fraud_rate:.3%}")
        print(f"   International Risk Multiplier: {international_fraud_rate/domestic_fraud_rate:.1f}x")
        
        # Time-based fraud patterns
        print(f"\n⏰ Time-based Fraud Patterns:")
        transactions['hour'] = pd.to_datetime(transactions['transaction_time'], format='%H:%M:%S').dt.hour
        
        # Fraud by hour of day
        hourly_fraud = transactions.groupby('hour')['is_fraud'].mean()
        peak_fraud_hours = hourly_fraud.nlargest(3)
        print(f"   Peak fraud hours:")
        for hour, rate in peak_fraud_hours.items():
            print(f"     {hour:02d}:00: {rate:.3%}")
        
        # Weekend vs weekday
        transactions['weekday'] = transactions['transaction_date'].dt.weekday
        weekend_fraud_rate = transactions[transactions['weekday'] >= 5]['is_fraud'].mean()
        weekday_fraud_rate = transactions[transactions['weekday'] < 5]['is_fraud'].mean()
        
        print(f"   Weekday fraud rate: {weekday_fraud_rate:.3%}")
        print(f"   Weekend fraud rate: {weekend_fraud_rate:.3%}")
        
        # Customer risk profile analysis
        print(f"\n👤 Customer Risk Profile Analysis:")
        customer_fraud = transactions.groupby('risk_profile')['is_fraud'].agg(['count', 'sum', 'mean'])
        customer_fraud.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
        
        for risk_level, row in customer_fraud.iterrows():
            print(f"   {risk_level} Risk: {row['Fraud_Rate']:.3%} ({row['Fraud_Count']:.0f}/{row['Total_Transactions']:.0f})")
        
        # Store results
        fraud_analysis = {
            'total_transactions': int(total_transactions),
            'fraud_count': int(fraud_count),
            'fraud_rate': float(fraud_rate),
            'category_fraud_rates': category_fraud.to_dict('index'),
            'location_analysis': {
                'domestic_fraud_rate': float(domestic_fraud_rate),
                'international_fraud_rate': float(international_fraud_rate)
            },
            'peak_fraud_hours': peak_fraud_hours.to_dict(),
            'risk_profile_analysis': customer_fraud.to_dict('index')
        }
        
        self.analysis_results['fraud_analysis'] = fraud_analysis
        return fraud_analysis
    
    def analyze_credit_risk(self):
        """Analyze loan default and credit risk patterns"""
        print("\n🏦 CREDIT RISK ANALYSIS")
        print("-" * 50)
        
        loans = self.data['loans']
        credit_analysis = {}
        
        # Basic credit statistics
        total_applications = len(loans)
        approved_loans = loans[loans['is_approved'] == 1]
        approval_rate = len(approved_loans) / total_applications
        
        default_loans = approved_loans[approved_loans['is_default'] == 1]
        default_rate = len(default_loans) / len(approved_loans) if len(approved_loans) > 0 else 0
        
        print(f"📊 Basic Credit Statistics:")
        print(f"   Total Applications: {total_applications:,}")
        print(f"   Approved Loans: {len(approved_loans):,}")
        print(f"   Approval Rate: {approval_rate:.1%}")
        print(f"   Default Rate: {default_rate:.1%}")
        
        # Default risk by credit score ranges
        print(f"\n📈 Default Risk by Credit Score:")
        score_ranges = [(300, 580), (580, 669), (670, 739), (740, 799), (800, 850)]
        
        for low, high in score_ranges:
            mask = (approved_loans['credit_score'] >= low) & (approved_loans['credit_score'] <= high)
            score_group = approved_loans[mask]
            default_rate_group = score_group['is_default'].mean() if len(score_group) > 0 else 0
            
            if low == 300:
                range_label = f"{low}-{high} (Poor)"
            elif low == 580:
                range_label = f"{low}-{high} (Fair)"
            elif low == 670:
                range_label = f"{low}-{high} (Good)"
            elif low == 740:
                range_label = f"{low}-{high} (Very Good)"
            else:
                range_label = f"{low}-{high} (Excellent)"
            
            print(f"   {range_label}: {default_rate_group:.1%} default rate ({len(score_group):,} loans)")
        
        # Default risk by loan type
        print(f"\n🏠 Default Risk by Loan Type:")
        loan_type_analysis = approved_loans.groupby('loan_type')['is_default'].agg(['count', 'sum', 'mean'])
        loan_type_analysis.columns = ['Total_Loans', 'Defaults', 'Default_Rate']
        loan_type_analysis = loan_type_analysis.sort_values('Default_Rate', ascending=False)
        
        for loan_type, row in loan_type_analysis.iterrows():
            print(f"   {loan_type}: {row['Default_Rate']:.1%} ({row['Defaults']:.0f}/{row['Total_Loans']:.0f})")
        
        # Default risk by income levels
        print(f"\n💰 Default Risk by Income Level:")
        approved_loans['income_bracket'] = pd.cut(approved_loans['annual_income'], 
                                                bins=[0, 50000, 100000, 150000, float('inf')],
                                                labels=['<$50K', '$50K-$100K', '$100K-$150K', '$150K+'])
        
        income_analysis = approved_loans.groupby('income_bracket')['is_default'].agg(['count', 'sum', 'mean'])
        income_analysis.columns = ['Total_Loans', 'Defaults', 'Default_Rate']
        
        for income_bracket, row in income_analysis.iterrows():
            print(f"   {income_bracket}: {row['Default_Rate']:.1%} ({row['Defaults']:.0f}/{row['Total_Loans']:.0f})")
        
        # Default risk by debt-to-income ratio
        print(f"\n📊 Default Risk by Debt-to-Income Ratio:")
        approved_loans['dti_bracket'] = pd.cut(approved_loans['debt_to_income_ratio'],
                                             bins=[0, 0.2, 0.3, 0.4, 1.0],
                                             labels=['<20%', '20-30%', '30-40%', '40%+'])
        
        dti_analysis = approved_loans.groupby('dti_bracket')['is_default'].agg(['count', 'sum', 'mean'])
        dti_analysis.columns = ['Total_Loans', 'Defaults', 'Default_Rate']
        
        for dti_bracket, row in dti_analysis.iterrows():
            print(f"   {dti_bracket}: {row['Default_Rate']:.1%} ({row['Defaults']:.0f}/{row['Total_Loans']:.0f})")
        
        # Average loan amounts and interest rates
        print(f"\n💵 Loan Economics:")
        avg_loan_amount = approved_loans['loan_amount'].mean()
        avg_interest_rate = approved_loans['interest_rate'].mean()
        
        print(f"   Average Loan Amount: ${avg_loan_amount:,.0f}")
        print(f"   Average Interest Rate: {avg_interest_rate:.2%}")
        
        # High-risk loan identification
        high_risk_loans = approved_loans[
            (approved_loans['credit_score'] < 650) | 
            (approved_loans['debt_to_income_ratio'] > 0.4) |
            (approved_loans['employment_years'] < 2)
        ]
        high_risk_default_rate = high_risk_loans['is_default'].mean()
        
        print(f"\n⚠️  High-Risk Loan Analysis:")
        print(f"   High-Risk Loans: {len(high_risk_loans):,} ({len(high_risk_loans)/len(approved_loans):.1%})")
        print(f"   High-Risk Default Rate: {high_risk_default_rate:.1%}")
        
        # Store results
        credit_analysis = {
            'total_applications': int(total_applications),
            'approval_rate': float(approval_rate),
            'default_rate': float(default_rate),
            'loan_type_analysis': loan_type_analysis.to_dict('index'),
            'credit_score_analysis': {},
            'high_risk_loans': {
                'count': int(len(high_risk_loans)),
                'percentage': float(len(high_risk_loans)/len(approved_loans)),
                'default_rate': float(high_risk_default_rate)
            }
        }
        
        self.analysis_results['credit_analysis'] = credit_analysis
        return credit_analysis
    
    def analyze_account_risk(self):
        """Analyze bank account risk patterns"""
        print("\n🏛️ ACCOUNT RISK ANALYSIS")
        print("-" * 50)
        
        accounts = self.data['accounts']
        account_analysis = {}
        
        # Basic account statistics
        total_accounts = len(accounts)
        suspicious_accounts = accounts[accounts['suspicious_activity'] == 1]
        suspicious_rate = len(suspicious_accounts) / total_accounts
        
        print(f"📊 Basic Account Statistics:")
        print(f"   Total Accounts: {total_accounts:,}")
        print(f"   Suspicious Accounts: {len(suspicious_accounts):,}")
        print(f"   Suspicious Activity Rate: {suspicious_rate:.1%}")
        
        # Risk by account type
        print(f"\n🏦 Risk by Account Type:")
        account_type_risk = accounts.groupby('account_type')['suspicious_activity'].agg(['count', 'sum', 'mean'])
        account_type_risk.columns = ['Total_Accounts', 'Suspicious_Count', 'Suspicious_Rate']
        account_type_risk = account_type_risk.sort_values('Suspicious_Rate', ascending=False)
        
        for account_type, row in account_type_risk.iterrows():
            print(f"   {account_type}: {row['Suspicious_Rate']:.1%} ({row['Suspicious_Count']:.0f}/{row['Total_Accounts']:.0f})")
        
        # Balance analysis
        print(f"\n💰 Balance Analysis:")
        avg_balance = accounts['current_balance'].mean()
        median_balance = accounts['current_balance'].median()
        
        print(f"   Average Balance: ${avg_balance:,.0f}")
        print(f"   Median Balance: ${median_balance:,.0f}")
        
        # High-value accounts
        high_value_threshold = accounts['current_balance'].quantile(0.95)  # Top 5%
        high_value_accounts = accounts[accounts['current_balance'] >= high_value_threshold]
        high_value_suspicious_rate = high_value_accounts['suspicious_activity'].mean()
        
        print(f"   High-Value Accounts (>${high_value_threshold:,.0f}+): {len(high_value_accounts):,}")
        print(f"   High-Value Suspicious Rate: {high_value_suspicious_rate:.1%}")
        
        # Transaction volume analysis
        print(f"\n📈 Transaction Volume Analysis:")
        high_volume_accounts = accounts[accounts['monthly_transactions'] > accounts['monthly_transactions'].quantile(0.9)]
        high_volume_suspicious_rate = high_volume_accounts['suspicious_activity'].mean()
        
        print(f"   High-Volume Accounts (>90th percentile): {len(high_volume_accounts):,}")
        print(f"   High-Volume Suspicious Rate: {high_volume_suspicious_rate:.1%}")
        
        # Overdraft analysis
        overdraft_accounts = accounts[accounts['overdraft_count_3m'] > 0]
        overdraft_suspicious_rate = overdraft_accounts['suspicious_activity'].mean()
        
        print(f"\n🚫 Overdraft Analysis:")
        print(f"   Accounts with Overdrafts: {len(overdraft_accounts):,} ({len(overdraft_accounts)/total_accounts:.1%})")
        print(f"   Overdraft Account Suspicious Rate: {overdraft_suspicious_rate:.1%}")
        
        # Account status analysis
        print(f"\n📋 Account Status Analysis:")
        status_analysis = accounts.groupby('account_status')['suspicious_activity'].agg(['count', 'sum', 'mean'])
        status_analysis.columns = ['Total_Accounts', 'Suspicious_Count', 'Suspicious_Rate']
        
        for status, row in status_analysis.iterrows():
            print(f"   {status}: {row['Suspicious_Rate']:.1%} ({row['Suspicious_Count']:.0f}/{row['Total_Accounts']:.0f})")
        
        # Store results
        account_analysis = {
            'total_accounts': int(total_accounts),
            'suspicious_rate': float(suspicious_rate),
            'account_type_risk': account_type_risk.to_dict('index'),
            'high_value_analysis': {
                'threshold': float(high_value_threshold),
                'count': int(len(high_value_accounts)),
                'suspicious_rate': float(high_value_suspicious_rate)
            }
        }
        
        self.analysis_results['account_analysis'] = account_analysis
        return account_analysis
    
    def analyze_economic_trends(self):
        """Analyze economic indicators and market trends"""
        print("\n📈 ECONOMIC TRENDS ANALYSIS")
        print("-" * 50)
        
        indicators = self.data['indicators']
        economic_analysis = {}
        
        # Recent economic conditions
        latest_data = indicators.iloc[-1]
        year_ago_data = indicators.iloc[-13] if len(indicators) >= 13 else indicators.iloc[0]
        
        print(f"📊 Current Economic Conditions (Latest Month):")
        print(f"   Unemployment Rate: {latest_data['unemployment_rate']:.1f}%")
        print(f"   GDP Growth Rate: {latest_data['gdp_growth_rate']:.1f}%")
        print(f"   Inflation Rate: {latest_data['inflation_rate']:.1f}%")
        print(f"   Federal Funds Rate: {latest_data['federal_funds_rate']:.2f}%")
        print(f"   Mortgage Rate: {latest_data['mortgage_rate']:.2f}%")
        
        # Year-over-year changes
        print(f"\n📈 Year-over-Year Changes:")
        unemployment_change = latest_data['unemployment_rate'] - year_ago_data['unemployment_rate']
        gdp_change = latest_data['gdp_growth_rate'] - year_ago_data['gdp_growth_rate']
        inflation_change = latest_data['inflation_rate'] - year_ago_data['inflation_rate']
        
        print(f"   Unemployment: {'↑' if unemployment_change > 0 else '↓'} {abs(unemployment_change):.1f} percentage points")
        print(f"   GDP Growth: {'↑' if gdp_change > 0 else '↓'} {abs(gdp_change):.1f} percentage points")
        print(f"   Inflation: {'↑' if inflation_change > 0 else '↓'} {abs(inflation_change):.1f} percentage points")
        
        # Market volatility
        print(f"\n📊 Market Conditions:")
        avg_vix = indicators['vix_volatility'].tail(12).mean()  # Last 12 months
        current_vix = latest_data['vix_volatility']
        
        print(f"   Current VIX: {current_vix:.1f}")
        print(f"   12-Month Avg VIX: {avg_vix:.1f}")
        
        if current_vix > 30:
            volatility_level = "HIGH"
        elif current_vix > 20:
            volatility_level = "MODERATE"
        else:
            volatility_level = "LOW"
        
        print(f"   Market Volatility: {volatility_level}")
        
        # Economic risk assessment
        print(f"\n⚠️  Economic Risk Assessment:")
        
        risk_factors = []
        if latest_data['unemployment_rate'] > 7:
            risk_factors.append("High unemployment")
        if latest_data['inflation_rate'] > 4:
            risk_factors.append("High inflation")
        if latest_data['gdp_growth_rate'] < 0:
            risk_factors.append("Economic contraction")
        if current_vix > 30:
            risk_factors.append("High market volatility")
        
        if len(risk_factors) == 0:
            print("   Economic Risk Level: LOW")
        elif len(risk_factors) <= 2:
            print("   Economic Risk Level: MODERATE")
        else:
            print("   Economic Risk Level: HIGH")
        
        for factor in risk_factors:
            print(f"     • {factor}")
        
        # Store results
        economic_analysis = {
            'current_conditions': latest_data.to_dict(),
            'risk_level': 'HIGH' if len(risk_factors) > 2 else ('MODERATE' if len(risk_factors) > 0 else 'LOW'),
            'risk_factors': risk_factors,
            'volatility_level': volatility_level
        }
        
        self.analysis_results['economic_analysis'] = economic_analysis
        return economic_analysis
    
    def generate_risk_insights(self):
        """Generate comprehensive risk insights and recommendations"""
        print("\n🎯 COMPREHENSIVE RISK INSIGHTS")
        print("-" * 50)
        
        insights = []
        recommendations = []
        
        # Fraud insights
        if 'fraud_analysis' in self.analysis_results:
            fraud_data = self.analysis_results['fraud_analysis']
            
            if fraud_data['fraud_rate'] > 0.005:  # > 0.5%
                insights.append(f"⚠️  Elevated fraud rate: {fraud_data['fraud_rate']:.3%}")
                recommendations.append("Implement enhanced fraud monitoring for high-risk categories")
            
            # International transaction risk
            if 'location_analysis' in fraud_data:
                intl_risk = fraud_data['location_analysis']['international_fraud_rate']
                if intl_risk > 0.01:  # > 1%
                    insights.append(f"🌍 High international fraud risk: {intl_risk:.3%}")
                    recommendations.append("Strengthen international transaction verification")
        
        # Credit risk insights
        if 'credit_analysis' in self.analysis_results:
            credit_data = self.analysis_results['credit_analysis']
            
            if credit_data['default_rate'] > 0.1:  # > 10%
                insights.append(f"🏦 Elevated default rate: {credit_data['default_rate']:.1%}")
                recommendations.append("Tighten credit approval criteria")
            
            if 'high_risk_loans' in credit_data:
                high_risk_pct = credit_data['high_risk_loans']['percentage']
                if high_risk_pct > 0.2:  # > 20%
                    insights.append(f"⚠️  High proportion of risky loans: {high_risk_pct:.1%}")
                    recommendations.append("Review underwriting standards for high-risk profiles")
        
        # Account risk insights
        if 'account_analysis' in self.analysis_results:
            account_data = self.analysis_results['account_analysis']
            
            if account_data['suspicious_rate'] > 0.05:  # > 5%
                insights.append(f"🚨 High suspicious activity rate: {account_data['suspicious_rate']:.1%}")
                recommendations.append("Enhance account monitoring and KYC procedures")
        
        # Economic insights
        if 'economic_analysis' in self.analysis_results:
            econ_data = self.analysis_results['economic_analysis']
            
            if econ_data['risk_level'] in ['HIGH', 'MODERATE']:
                insights.append(f"📈 Economic risk level: {econ_data['risk_level']}")
                recommendations.append("Adjust risk appetites based on economic conditions")
        
        print("🔍 Key Risk Insights:")
        for insight in insights:
            print(f"   {insight}")
        
        print(f"\n💡 Strategic Recommendations:")
        for recommendation in recommendations:
            print(f"   • {recommendation}")
        
        # Overall risk score calculation
        risk_factors = len(insights)
        if risk_factors >= 4:
            overall_risk = "HIGH"
        elif risk_factors >= 2:
            overall_risk = "MODERATE"
        else:
            overall_risk = "LOW"
        
        print(f"\n🎯 OVERALL RISK ASSESSMENT: {overall_risk}")
        
        # Store insights
        risk_insights = {
            'insights': insights,
            'recommendations': recommendations,
            'overall_risk_level': overall_risk,
            'risk_factor_count': risk_factors
        }
        
        self.analysis_results['risk_insights'] = risk_insights
        return risk_insights
    
    def save_analysis_results(self):
        """Save all analysis results to files"""
        os.makedirs('reports', exist_ok=True)
        
        # Save comprehensive analysis results
        with open('reports/financial_risk_analysis.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Create executive summary
        executive_summary = {
            'analysis_date': datetime.now().isoformat(),
            'overall_risk_level': self.analysis_results.get('risk_insights', {}).get('overall_risk_level', 'UNKNOWN'),
            'key_metrics': {
                'fraud_rate': self.analysis_results.get('fraud_analysis', {}).get('fraud_rate', 0),
                'default_rate': self.analysis_results.get('credit_analysis', {}).get('default_rate', 0),
                'suspicious_activity_rate': self.analysis_results.get('account_analysis', {}).get('suspicious_rate', 0)
            },
            'top_recommendations': self.analysis_results.get('risk_insights', {}).get('recommendations', [])[:3]
        }
        
        with open('reports/executive_summary.json', 'w') as f:
            json.dump(executive_summary, f, indent=2)
        
        print(f"\n💾 Analysis results saved:")
        print(f"   📄 reports/financial_risk_analysis.json")
        print(f"   📄 reports/executive_summary.json")
    
    def run_comprehensive_analysis(self):
        """Run complete financial risk analysis"""
        print("🚀 Starting Comprehensive Financial Risk Analysis...")
        
        # Load datasets
        if not self.load_all_datasets():
            return False
        
        # Run all analysis modules
        print("\n" + "="*80)
        self.analyze_fraud_patterns()
        
        print("\n" + "="*80)
        self.analyze_credit_risk()
        
        print("\n" + "="*80)
        self.analyze_account_risk()
        
        print("\n" + "="*80)
        self.analyze_economic_trends()
        
        print("\n" + "="*80)
        self.generate_risk_insights()
        
        # Save results
        self.save_analysis_results()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)
        print("📋 Analysis Summary:")
        
        if 'risk_insights' in self.analysis_results:
            insights = self.analysis_results['risk_insights']
            print(f"   🎯 Overall Risk Level: {insights['overall_risk_level']}")
            print(f"   🔍 Risk Factors Identified: {insights['risk_factor_count']}")
            print(f"   💡 Recommendations Generated: {len(insights['recommendations'])}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Run financial_risk_models.py to build ML models")
        print(f"   2. Run financial_risk_dashboards.py for visualizations")
        print(f"   3. Review reports/ folder for detailed analysis")
        print("="*80)
        
        return True

def main():
    """Main function to run comprehensive analysis"""
    analyzer = FinancialRiskAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("✅ Financial risk analysis completed successfully!")
    else:
        print("❌ Analysis failed. Please check data files.")
        
    return success

if __name__ == "__main__":
    main()