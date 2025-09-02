"""
Financial Risk Assessment & Fraud Detection System
Data Generation Module - Creates realistic financial datasets for analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏦 FINANCIAL RISK ASSESSMENT & FRAUD DETECTION SYSTEM")
print("=" * 80)
print("📊 DATA GENERATION MODULE")
print("-" * 40)

class FinancialDataGenerator:
    """Generate realistic financial datasets for risk assessment"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.setup_directories()
        
    def setup_directories(self):
        """Create project directory structure"""
        directories = [
            'data', 'models', 'dashboards', 'reports', 
            'api', 'notebooks', 'documentation', 'config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("✅ Project structure created")
    
    def generate_transaction_data(self, n_transactions=50000):
        """Generate credit card transaction dataset"""
        print(f"📊 Generating {n_transactions:,} credit card transactions...")
        
        # Customer demographics
        n_customers = 8000
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        
        # Generate customer profiles
        customers = pd.DataFrame({
            'customer_id': customer_ids,
            'age': np.random.randint(18, 80, n_customers),
            'income': np.random.lognormal(10.5, 0.5, n_customers).astype(int),
            'credit_score': np.random.normal(680, 120, n_customers).astype(int),
            'account_age_months': np.random.randint(1, 240, n_customers),
            'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_customers),
            'risk_profile': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n_customers, p=[0.7, 0.25, 0.05])
        })
        
        # Clip credit scores to realistic range
        customers['credit_score'] = np.clip(customers['credit_score'], 300, 850)
        
        # Generate transactions
        transactions = []
        fraud_rate = 0.002  # 0.2% fraud rate (realistic)
        
        # Transaction categories and typical amounts
        categories = {
            'Grocery': (20, 200),
            'Gas Station': (25, 100),
            'Restaurant': (15, 150),
            'Online Retail': (20, 300),
            'Department Store': (30, 500),
            'ATM Withdrawal': (20, 500),
            'Hotel': (100, 800),
            'Airline': (200, 1500),
            'Medical': (50, 2000),
            'Electronics': (100, 3000),
            'Jewelry': (200, 5000),
            'Cash Advance': (100, 1000)
        }
        
        for i in range(n_transactions):
            # Select random customer
            customer = customers.sample(1).iloc[0]
            
            # Generate transaction details
            category = np.random.choice(list(categories.keys()))
            amount_range = categories[category]
            
            # Base amount from category range
            if customer['income'] > 100000:
                multiplier = np.random.uniform(1.2, 2.0)
            elif customer['income'] > 50000:
                multiplier = np.random.uniform(0.8, 1.5)
            else:
                multiplier = np.random.uniform(0.5, 1.0)
                
            base_amount = np.random.uniform(amount_range[0], amount_range[1])
            amount = base_amount * multiplier
            
            # Time-based patterns
            days_ago = np.random.exponential(30)  # More recent transactions
            days_ago = min(days_ago, 365)
            transaction_time = datetime.now() - timedelta(days=days_ago)
            
            # Hour patterns (more transactions during day)
            if np.random.random() < 0.7:  # 70% daytime transactions
                hour = np.random.normal(14, 4)  # Peak at 2 PM
            else:
                hour = np.random.uniform(0, 24)  # Random time
            hour = max(0, min(23, int(hour)))
            
            transaction_time = transaction_time.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Location patterns
            if np.random.random() < 0.95:  # 95% domestic
                country = 'USA'
                if np.random.random() < 0.8:  # 80% same state
                    state = customer['state']
                else:
                    state = np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'])
            else:  # 5% international
                country = np.random.choice(['Canada', 'Mexico', 'UK', 'France', 'Japan'])
                state = 'N/A'
            
            # Determine if fraudulent
            is_fraud = 0
            fraud_probability = fraud_rate
            
            # Increase fraud probability for certain conditions
            if amount > 1000:
                fraud_probability *= 3
            if country != 'USA':
                fraud_probability *= 5
            if hour < 6 or hour > 22:  # Late night transactions
                fraud_probability *= 2
            if category in ['Electronics', 'Jewelry']:
                fraud_probability *= 2
                
            is_fraud = np.random.random() < fraud_probability
            
            # If fraud, modify transaction characteristics
            if is_fraud:
                # Fraudulent transactions tend to be higher amounts
                amount *= np.random.uniform(1.5, 4.0)
                # More likely to be international
                if np.random.random() < 0.3:
                    country = np.random.choice(['Russia', 'Nigeria', 'Romania', 'Philippines'])
                    state = 'N/A'
                # More likely to be electronics/high-value items
                if np.random.random() < 0.4:
                    category = np.random.choice(['Electronics', 'Jewelry', 'Cash Advance'])
            
            # Generate merchant information
            merchant_id = f"MERCH_{np.random.randint(1, 10000):05d}"
            
            transaction = {
                'transaction_id': f"TXN_{i+1:08d}",
                'customer_id': customer['customer_id'],
                'transaction_date': transaction_time.strftime('%Y-%m-%d'),
                'transaction_time': transaction_time.strftime('%H:%M:%S'),
                'amount': round(amount, 2),
                'category': category,
                'merchant_id': merchant_id,
                'country': country,
                'state': state,
                'is_fraud': int(is_fraud),
                'customer_age': customer['age'],
                'customer_income': customer['income'],
                'customer_credit_score': customer['credit_score'],
                'account_age_months': customer['account_age_months'],
                'risk_profile': customer['risk_profile']
            }
            
            transactions.append(transaction)
        
        # Create DataFrames
        transactions_df = pd.DataFrame(transactions)
        
        # Save datasets
        transactions_df.to_csv('data/credit_card_transactions.csv', index=False)
        customers.to_csv('data/customer_profiles.csv', index=False)
        
        print(f"✅ Generated credit_card_transactions.csv: {transactions_df.shape}")
        print(f"✅ Generated customer_profiles.csv: {customers.shape}")
        print(f"📊 Fraud rate: {transactions_df['is_fraud'].mean():.3%}")
        
        return transactions_df, customers
    
    def generate_loan_data(self, n_loans=15000):
        """Generate loan application and default dataset"""
        print(f"📊 Generating {n_loans:,} loan applications...")
        
        # Loan types and characteristics
        loan_types = {
            'Personal': {'min_amount': 5000, 'max_amount': 50000, 'term_months': [24, 36, 48, 60]},
            'Auto': {'min_amount': 15000, 'max_amount': 80000, 'term_months': [36, 48, 60, 72]},
            'Mortgage': {'min_amount': 100000, 'max_amount': 800000, 'term_months': [180, 240, 300, 360]},
            'Business': {'min_amount': 25000, 'max_amount': 500000, 'term_months': [24, 36, 48, 60, 84]},
            'Student': {'min_amount': 10000, 'max_amount': 100000, 'term_months': [120, 180, 240]}
        }
        
        loans = []
        default_rate = 0.08  # 8% default rate
        
        for i in range(n_loans):
            # Applicant characteristics
            age = max(18, int(np.random.normal(40, 15)))
            income = max(20000, int(np.random.lognormal(10.8, 0.6)))
            
            # Credit score influences approval and terms
            base_credit_score = np.random.normal(680, 120)
            credit_score = max(300, min(850, int(base_credit_score)))
            
            # Employment and financial history
            employment_years = max(0, int(np.random.exponential(5)))
            debt_to_income = np.random.beta(2, 5)  # Most people have lower DTI
            
            # Select loan type based on demographics
            if age < 30:
                loan_type_probs = [0.3, 0.25, 0.1, 0.05, 0.3]  # More personal/student loans
            elif age < 50:
                loan_type_probs = [0.2, 0.3, 0.35, 0.1, 0.05]  # More auto/mortgage
            else:
                loan_type_probs = [0.25, 0.15, 0.4, 0.15, 0.05]  # More mortgage/business
                
            loan_type = np.random.choice(list(loan_types.keys()), p=loan_type_probs)
            loan_config = loan_types[loan_type]
            
            # Loan amount based on income and loan type
            max_affordable = income * 0.4 * 12  # 40% of annual income
            loan_amount = np.random.uniform(
                loan_config['min_amount'], 
                min(loan_config['max_amount'], max_affordable)
            )
            
            term_months = np.random.choice(loan_config['term_months'])
            
            # Interest rate based on credit score and loan type
            base_rates = {
                'Personal': 0.12, 'Auto': 0.06, 'Mortgage': 0.04, 
                'Business': 0.08, 'Student': 0.05
            }
            
            base_rate = base_rates[loan_type]
            credit_adjustment = (750 - credit_score) / 1000  # Better credit = lower rate
            interest_rate = max(0.03, base_rate + credit_adjustment + np.random.normal(0, 0.01))
            
            # Approval decision
            approval_score = (
                (credit_score - 300) / 550 * 0.4 +  # Credit score weight
                min(income / 100000, 1) * 0.3 +      # Income weight
                (1 - debt_to_income) * 0.2 +         # DTI weight
                min(employment_years / 10, 1) * 0.1  # Employment weight
            )
            
            is_approved = approval_score > 0.5 and np.random.random() < approval_score
            
            # Default probability (only for approved loans)
            default_probability = default_rate
            if is_approved:
                # Adjust default probability based on risk factors
                if credit_score < 600:
                    default_probability *= 3
                if debt_to_income > 0.4:
                    default_probability *= 2
                if employment_years < 2:
                    default_probability *= 1.5
                if loan_amount > income * 5:
                    default_probability *= 2
                    
                is_default = np.random.random() < default_probability
            else:
                is_default = False
            
            # Application date (last 2 years)
            app_date = datetime.now() - timedelta(days=np.random.randint(1, 730))
            
            loan = {
                'loan_id': f"LOAN_{i+1:07d}",
                'application_date': app_date.strftime('%Y-%m-%d'),
                'applicant_age': age,
                'annual_income': income,
                'credit_score': credit_score,
                'employment_years': employment_years,
                'debt_to_income_ratio': round(debt_to_income, 3),
                'loan_type': loan_type,
                'loan_amount': round(loan_amount, 2),
                'loan_term_months': term_months,
                'interest_rate': round(interest_rate, 4),
                'is_approved': int(is_approved),
                'is_default': int(is_default),
                'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
                'purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'business', 
                                          'education', 'medical', 'vacation', 'other']),
                'own_home': int(np.random.random() < 0.65),
                'years_at_address': max(0, int(np.random.exponential(4)))
            }
            
            loans.append(loan)
        
        loans_df = pd.DataFrame(loans)
        loans_df.to_csv('data/loan_applications.csv', index=False)
        
        print(f"✅ Generated loan_applications.csv: {loans_df.shape}")
        print(f"📊 Approval rate: {loans_df['is_approved'].mean():.1%}")
        print(f"📊 Default rate (approved loans): {loans_df[loans_df['is_approved']==1]['is_default'].mean():.1%}")
        
        return loans_df
    
    def generate_bank_accounts_data(self, n_accounts=25000):
        """Generate bank account and transaction data"""
        print(f"📊 Generating {n_accounts:,} bank accounts...")
        
        # Account types and characteristics
        account_types = ['Checking', 'Savings', 'Money Market', 'CD', 'Business Checking']
        
        accounts = []
        for i in range(n_accounts):
            # Customer demographics
            age = max(18, int(np.random.normal(45, 18)))
            
            # Account characteristics
            account_type = np.random.choice(account_types, p=[0.4, 0.3, 0.15, 0.1, 0.05])
            open_date = datetime.now() - timedelta(days=np.random.randint(30, 3650))
            
            # Balance based on account type and customer profile
            if account_type == 'Checking':
                balance = max(0, np.random.lognormal(8, 1.5))
            elif account_type == 'Savings':
                balance = max(0, np.random.lognormal(9, 1.2))
            elif account_type == 'Money Market':
                balance = max(0, np.random.lognormal(10, 1))
            elif account_type == 'CD':
                balance = max(0, np.random.lognormal(10.5, 0.8))
            else:  # Business
                balance = max(0, np.random.lognormal(11, 1.5))
            
            # Transaction patterns (monthly averages)
            if account_type == 'Checking':
                monthly_transactions = max(1, int(np.random.poisson(25)))
                monthly_volume = balance * np.random.uniform(1.5, 4.0)
            elif account_type == 'Business Checking':
                monthly_transactions = max(1, int(np.random.poisson(45)))
                monthly_volume = balance * np.random.uniform(2.0, 6.0)
            else:
                monthly_transactions = max(1, int(np.random.poisson(8)))
                monthly_volume = balance * np.random.uniform(0.2, 0.8)
            
            # Risk indicators
            avg_daily_balance = balance * np.random.uniform(0.7, 1.3)
            overdraft_count = max(0, int(np.random.poisson(0.5)))  # Average 0.5 overdrafts per month
            
            # Suspicious activity flags
            suspicious_activity = 0
            if monthly_transactions > 100 or monthly_volume > balance * 10:
                suspicious_activity = np.random.choice([0, 1], p=[0.7, 0.3])
            
            account = {
                'account_id': f"ACC_{i+1:08d}",
                'customer_age': age,
                'account_type': account_type,
                'account_open_date': open_date.strftime('%Y-%m-%d'),
                'current_balance': round(balance, 2),
                'avg_daily_balance': round(avg_daily_balance, 2),
                'monthly_transactions': monthly_transactions,
                'monthly_volume': round(monthly_volume, 2),
                'overdraft_count_3m': overdraft_count,
                'credit_score': max(300, min(850, int(np.random.normal(700, 100)))),
                'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
                'suspicious_activity': suspicious_activity,
                'account_status': np.random.choice(['Active', 'Dormant', 'Closed'], p=[0.85, 0.12, 0.03])
            }
            
            accounts.append(account)
        
        accounts_df = pd.DataFrame(accounts)
        accounts_df.to_csv('data/bank_accounts.csv', index=False)
        
        print(f"✅ Generated bank_accounts.csv: {accounts_df.shape}")
        print(f"📊 Suspicious activity rate: {accounts_df['suspicious_activity'].mean():.1%}")
        
        return accounts_df
    
    def generate_economic_indicators(self):
        """Generate economic indicators dataset"""
        print("📊 Generating economic indicators...")
        
        # Generate 5 years of monthly data
        dates = pd.date_range(start='2019-01-01', end='2024-12-31', freq='M')
        
        indicators = []
        base_unemployment = 5.0
        base_gdp_growth = 2.5
        base_inflation = 2.0
        
        for i, date in enumerate(dates):
            # Add trends and seasonality
            unemployment_rate = base_unemployment + np.sin(i/6) * 0.5 + np.random.normal(0, 0.3)
            gdp_growth = base_gdp_growth + np.cos(i/12) * 0.8 + np.random.normal(0, 0.5)
            inflation_rate = base_inflation + np.sin(i/4) * 0.3 + np.random.normal(0, 0.2)
            
            # Interest rates
            federal_funds_rate = max(0.1, 2.0 + np.sin(i/8) * 1.5 + np.random.normal(0, 0.2))
            mortgage_rate = federal_funds_rate + 2.0 + np.random.normal(0, 0.1)
            
            # Market indicators
            stock_market_index = 3000 + i * 10 + np.random.normal(0, 50)
            vix_volatility = max(10, 20 + np.sin(i/3) * 5 + np.random.normal(0, 2))
            
            indicator = {
                'date': date.strftime('%Y-%m-%d'),
                'unemployment_rate': round(max(0, unemployment_rate), 2),
                'gdp_growth_rate': round(gdp_growth, 2),
                'inflation_rate': round(max(0, inflation_rate), 2),
                'federal_funds_rate': round(max(0, federal_funds_rate), 2),
                'mortgage_rate': round(max(0, mortgage_rate), 2),
                'stock_market_index': round(max(0, stock_market_index), 2),
                'vix_volatility': round(max(0, vix_volatility), 2),
                'consumer_confidence': round(max(0, 100 + np.random.normal(0, 10)), 1)
            }
            
            indicators.append(indicator)
        
        indicators_df = pd.DataFrame(indicators)
        indicators_df.to_csv('data/economic_indicators.csv', index=False)
        
        print(f"✅ Generated economic_indicators.csv: {indicators_df.shape}")
        
        return indicators_df
    
    def generate_all_datasets(self):
        """Generate all financial datasets"""
        print("🚀 Starting comprehensive financial data generation...")
        print("This will create realistic datasets for:")
        print("   • Credit card transactions (fraud detection)")
        print("   • Loan applications (credit risk assessment)")
        print("   • Bank accounts (account monitoring)")
        print("   • Economic indicators (market analysis)")
        print()
        
        # Generate all datasets
        transactions_df, customers_df = self.generate_transaction_data(50000)
        loans_df = self.generate_loan_data(15000)
        accounts_df = self.generate_bank_accounts_data(25000)
        indicators_df = self.generate_economic_indicators()
        
        # Generate summary statistics
        summary = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {
                'credit_card_transactions': {
                    'records': len(transactions_df),
                    'fraud_rate': float(transactions_df['is_fraud'].mean()),
                    'avg_transaction_amount': float(transactions_df['amount'].mean()),
                    'date_range': f"{transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}"
                },
                'loan_applications': {
                    'records': len(loans_df),
                    'approval_rate': float(loans_df['is_approved'].mean()),
                    'default_rate': float(loans_df[loans_df['is_approved']==1]['is_default'].mean()),
                    'avg_loan_amount': float(loans_df['loan_amount'].mean())
                },
                'bank_accounts': {
                    'records': len(accounts_df),
                    'suspicious_activity_rate': float(accounts_df['suspicious_activity'].mean()),
                    'avg_balance': float(accounts_df['current_balance'].mean())
                },
                'customer_profiles': {
                    'records': len(customers_df),
                    'avg_credit_score': float(customers_df['credit_score'].mean()),
                    'avg_income': float(customers_df['income'].mean())
                },
                'economic_indicators': {
                    'records': len(indicators_df),
                    'date_range': f"{indicators_df['date'].min()} to {indicators_df['date'].max()}"
                }
            }
        }
        
        import json
        with open('data/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("🎉 DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Generated datasets:")
        print("   📄 credit_card_transactions.csv - 50,000 transactions with fraud labels")
        print("   📄 loan_applications.csv - 15,000 loan applications with defaults")
        print("   📄 bank_accounts.csv - 25,000 bank accounts with risk indicators")
        print("   📄 customer_profiles.csv - 8,000 customer profiles")
        print("   📄 economic_indicators.csv - Economic data (2019-2024)")
        print("   📄 dataset_summary.json - Summary statistics")
        print()
        print("💡 Next steps:")
        print("   1. Run financial_risk_analysis.py for comprehensive analysis")
        print("   2. Run financial_risk_models.py to build ML models")
        print("   3. Run financial_risk_dashboards.py to create visualizations")
        print("="*80)
        
        return summary

def main():
    """Main function to generate all financial datasets"""
    print("Starting Financial Risk Assessment Data Generation...")
    
    generator = FinancialDataGenerator()
    summary = generator.generate_all_datasets()
    
    print("✅ All datasets generated successfully!")
    return summary

if __name__ == "__main__":
    main()