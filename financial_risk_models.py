"""
Financial Risk Assessment & Fraud Detection System
Machine Learning Models Module - Build and evaluate ML models for risk assessment
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏦 FINANCIAL RISK ASSESSMENT & FRAUD DETECTION SYSTEM")
print("=" * 80)
print("🤖 MACHINE LEARNING MODELS MODULE")
print("-" * 40)

class FinancialRiskModels:
    """Build and evaluate ML models for fraud detection and credit risk"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_results = {}
        
    def load_data(self):
        """Load financial datasets for modeling"""
        print("📂 Loading datasets for ML modeling...")
        
        try:
            self.transactions = pd.read_csv('data/credit_card_transactions.csv')
            self.customers = pd.read_csv('data/customer_profiles.csv')
            self.loans = pd.read_csv('data/loan_applications.csv')
            self.accounts = pd.read_csv('data/bank_accounts.csv')
            
            print(f"✅ Loaded transactions: {self.transactions.shape}")
            print(f"✅ Loaded customers: {self.customers.shape}")
            print(f"✅ Loaded loans: {self.loans.shape}")
            print(f"✅ Loaded accounts: {self.accounts.shape}")
            
            return True
        except FileNotFoundError as e:
            print(f"❌ Data file not found: {e}")
            print("Run financial_data_generator.py first to create datasets")
            return False
    
    def prepare_fraud_detection_data(self):
        """Prepare data for fraud detection models"""
        print("\n🔍 Preparing fraud detection dataset...")
        
        # Feature engineering for fraud detection
        transactions = self.transactions.copy()
        
        # Time-based features
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        transactions['transaction_time'] = pd.to_datetime(transactions['transaction_time'], format='%H:%M:%S')
        transactions['hour'] = transactions['transaction_time'].dt.hour
        transactions['day_of_week'] = transactions['transaction_date'].dt.dayofweek
        transactions['is_weekend'] = (transactions['day_of_week'] >= 5).astype(int)
        transactions['is_night'] = ((transactions['hour'] < 6) | (transactions['hour'] > 22)).astype(int)
        
        # Amount-based features
        transactions['log_amount'] = np.log1p(transactions['amount'])
        customer_stats = transactions.groupby('customer_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 'customer_transaction_count']
        transactions = transactions.merge(customer_stats, on='customer_id', how='left')
        
        # Amount deviation from customer's typical spending
        transactions['amount_deviation'] = (transactions['amount'] - transactions['customer_avg_amount']) / (transactions['customer_std_amount'] + 1)
        
        # Categorical encoding
        le_category = LabelEncoder()
        le_country = LabelEncoder()
        le_state = LabelEncoder()
        le_risk = LabelEncoder()
        
        transactions['category_encoded'] = le_category.fit_transform(transactions['category'])
        transactions['country_encoded'] = le_country.fit_transform(transactions['country'])
        transactions['state_encoded'] = le_state.fit_transform(transactions['state'])
        transactions['risk_profile_encoded'] = le_risk.fit_transform(transactions['risk_profile'])
        
        # Select features for fraud detection
        fraud_features = [
            'amount', 'log_amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'customer_age', 'customer_income', 'customer_credit_score', 'account_age_months',
            'category_encoded', 'country_encoded', 'state_encoded', 'risk_profile_encoded',
            'customer_avg_amount', 'customer_std_amount', 'customer_transaction_count',
            'amount_deviation'
        ]
        
        X_fraud = transactions[fraud_features].fillna(0)
        y_fraud = transactions['is_fraud']
        
        # Store encoders for later use
        self.encoders['fraud'] = {
            'category': le_category,
            'country': le_country,
            'state': le_state,
            'risk_profile': le_risk,
            'features': fraud_features
        }
        
        print(f"✅ Fraud detection dataset prepared: {X_fraud.shape}")
        print(f"   Features: {len(fraud_features)}")
        print(f"   Fraud rate: {y_fraud.mean():.3%}")
        
        return X_fraud, y_fraud
    
    def prepare_credit_risk_data(self):
        """Prepare data for credit risk models"""
        print("\n🏦 Preparing credit risk dataset...")
        
        # Use only approved loans for default prediction
        loans = self.loans[self.loans['is_approved'] == 1].copy()
        
        # Feature engineering for credit risk
        loans['application_date'] = pd.to_datetime(loans['application_date'])
        loans['loan_to_income_ratio'] = loans['loan_amount'] / loans['annual_income']
        loans['monthly_payment'] = loans['loan_amount'] / loans['loan_term_months']
        loans['payment_to_income_ratio'] = (loans['monthly_payment'] * 12) / loans['annual_income']
        
        # Categorical encoding
        le_loan_type = LabelEncoder()
        le_state = LabelEncoder()
        le_purpose = LabelEncoder()
        
        loans['loan_type_encoded'] = le_loan_type.fit_transform(loans['loan_type'])
        loans['state_encoded'] = le_state.fit_transform(loans['state'])
        loans['purpose_encoded'] = le_purpose.fit_transform(loans['purpose'])
        
        # Select features for credit risk
        credit_features = [
            'applicant_age', 'annual_income', 'credit_score', 'employment_years',
            'debt_to_income_ratio', 'loan_amount', 'loan_term_months', 'interest_rate',
            'loan_type_encoded', 'state_encoded', 'purpose_encoded', 'own_home',
            'years_at_address', 'loan_to_income_ratio', 'payment_to_income_ratio'
        ]
        
        X_credit = loans[credit_features].fillna(0)
        y_credit = loans['is_default']
        
        # Store encoders for later use
        self.encoders['credit'] = {
            'loan_type': le_loan_type,
            'state': le_state,
            'purpose': le_purpose,
            'features': credit_features
        }
        
        print(f"✅ Credit risk dataset prepared: {X_credit.shape}")
        print(f"   Features: {len(credit_features)}")
        print(f"   Default rate: {y_credit.mean():.3%}")
        
        return X_credit, y_credit
    
    def train_fraud_detection_models(self, X, y):
        """Train multiple fraud detection models"""
        print("\n🔍 Training fraud detection models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['fraud'] = scaler
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        
        fraud_results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Train model
            if name == 'SVM':
                # Use scaled features for SVM
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Cross-validation
            if name == 'SVM':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
            
            fraud_results[name] = results
            self.models[f'fraud_{name.lower().replace(" ", "_")}'] = model
            
            print(f"     Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, CV AUC: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Find best model
        best_model_name = max(fraud_results.keys(), key=lambda k: fraud_results[k]['auc'])
        print(f"\n🏆 Best fraud detection model: {best_model_name} (AUC: {fraud_results[best_model_name]['auc']:.3f})")
        
        self.model_results['fraud_detection'] = {
            'results': fraud_results,
            'best_model': best_model_name,
            'feature_names': self.encoders['fraud']['features']
        }
        
        return fraud_results
    
    def train_credit_risk_models(self, X, y):
        """Train multiple credit risk models"""
        print("\n🏦 Training credit risk models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['credit'] = scaler
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        
        credit_results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Train model
            if name == 'SVM':
                # Use scaled features for SVM
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Cross-validation
            if name == 'SVM':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
            
            credit_results[name] = results
            self.models[f'credit_{name.lower().replace(" ", "_")}'] = model
            
            print(f"     Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, CV AUC: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Find best model
        best_model_name = max(credit_results.keys(), key=lambda k: credit_results[k]['auc'])
        print(f"\n🏆 Best credit risk model: {best_model_name} (AUC: {credit_results[best_model_name]['auc']:.3f})")
        
        self.model_results['credit_risk'] = {
            'results': credit_results,
            'best_model': best_model_name,
            'feature_names': self.encoders['credit']['features']
        }
        
        return credit_results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        print("\n📊 Analyzing feature importance...")
        
        feature_importance = {}
        
        # Fraud detection feature importance
        if 'fraud_random_forest' in self.models:
            rf_fraud = self.models['fraud_random_forest']
            feature_names = self.encoders['fraud']['features']
            importance_scores = rf_fraud.feature_importances_
            
            fraud_importance = list(zip(feature_names, importance_scores))
            fraud_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("🔍 Top 10 Fraud Detection Features:")
            for i, (feature, importance) in enumerate(fraud_importance[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
            
            feature_importance['fraud_detection'] = fraud_importance
        
        # Credit risk feature importance
        if 'credit_random_forest' in self.models:
            rf_credit = self.models['credit_random_forest']
            feature_names = self.encoders['credit']['features']
            importance_scores = rf_credit.feature_importances_
            
            credit_importance = list(zip(feature_names, importance_scores))
            credit_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n🏦 Top 10 Credit Risk Features:")
            for i, (feature, importance) in enumerate(credit_importance[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
            
            feature_importance['credit_risk'] = credit_importance
        
        self.model_results['feature_importance'] = feature_importance
        return feature_importance
    
    def save_models_and_results(self):
        """Save trained models and results"""
        print("\n💾 Saving models and results...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = f'models/{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   Saved: {model_name}.pkl")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f'models/{scaler_name}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save encoders
        for encoder_name, encoder_dict in self.encoders.items():
            encoder_path = f'models/{encoder_name}_encoders.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder_dict, f)
        
        # Save results
        results_with_metadata = {
            'training_date': datetime.now().isoformat(),
            'model_results': self.model_results
        }
        
        with open('models/model_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # Create model deployment info
        deployment_info = {
            'fraud_detection': {
                'best_model_file': f'fraud_{self.model_results["fraud_detection"]["best_model"].lower().replace(" ", "_")}.pkl',
                'scaler_file': 'fraud_scaler.pkl',
                'encoders_file': 'fraud_encoders.pkl',
                'features': self.encoders['fraud']['features'],
                'performance': self.model_results['fraud_detection']['results'][self.model_results['fraud_detection']['best_model']]
            },
            'credit_risk': {
                'best_model_file': f'credit_{self.model_results["credit_risk"]["best_model"].lower().replace(" ", "_")}.pkl',
                'scaler_file': 'credit_scaler.pkl',
                'encoders_file': 'credit_encoders.pkl',
                'features': self.encoders['credit']['features'],
                'performance': self.model_results['credit_risk']['results'][self.model_results['credit_risk']['best_model']]
            }
        }
        
        with open('models/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print("✅ Models and results saved successfully!")
    
    def generate_model_insights(self):
        """Generate insights about model performance"""
        print("\n🎯 MODEL PERFORMANCE INSIGHTS")
        print("-" * 50)
        
        insights = []
        
        # Fraud detection insights
        if 'fraud_detection' in self.model_results:
            fraud_results = self.model_results['fraud_detection']['results']
            best_fraud_model = self.model_results['fraud_detection']['best_model']
            best_fraud_auc = fraud_results[best_fraud_model]['auc']
            
            print("🔍 Fraud Detection Performance:")
            print(f"   Best Model: {best_fraud_model}")
            print(f"   AUC Score: {best_fraud_auc:.3f}")
            
            if best_fraud_auc >= 0.95:
                insights.append("Excellent fraud detection performance - ready for production")
            elif best_fraud_auc >= 0.90:
                insights.append("Good fraud detection performance - consider additional tuning")
            else:
                insights.append("Moderate fraud detection performance - needs improvement")
            
            # Check model consistency
            cv_std = fraud_results[best_fraud_model]['cv_auc_std']
            if cv_std < 0.02:
                insights.append("Fraud models show consistent performance across validation folds")
            else:
                insights.append("Consider ensemble methods to improve fraud model stability")
        
        # Credit risk insights
        if 'credit_risk' in self.model_results:
            credit_results = self.model_results['credit_risk']['results']
            best_credit_model = self.model_results['credit_risk']['best_model']
            best_credit_auc = credit_results[best_credit_model]['auc']
            
            print(f"\n🏦 Credit Risk Assessment Performance:")
            print(f"   Best Model: {best_credit_model}")
            print(f"   AUC Score: {best_credit_auc:.3f}")
            
            if best_credit_auc >= 0.80:
                insights.append("Strong credit risk assessment capability")
            elif best_credit_auc >= 0.70:
                insights.append("Moderate credit risk assessment - acceptable for business use")
            else:
                insights.append("Credit risk models need significant improvement")
        
        # Feature importance insights
        if 'feature_importance' in self.model_results:
            print(f"\n📊 Feature Importance Insights:")
            
            # Fraud detection top features
            if 'fraud_detection' in self.model_results['feature_importance']:
                top_fraud_feature = self.model_results['feature_importance']['fraud_detection'][0]
                print(f"   Most important fraud indicator: {top_fraud_feature[0]}")
                insights.append(f"Focus fraud monitoring on {top_fraud_feature[0]} patterns")
            
            # Credit risk top features
            if 'credit_risk' in self.model_results['feature_importance']:
                top_credit_feature = self.model_results['feature_importance']['credit_risk'][0]
                print(f"   Most important credit risk factor: {top_credit_feature[0]}")
                insights.append(f"Prioritize {top_credit_feature[0]} in credit decisions")
        
        print(f"\n💡 Key Insights:")
        for insight in insights:
            print(f"   • {insight}")
        
        # Business impact estimation
        print(f"\n💰 Estimated Business Impact:")
        
        if 'fraud_detection' in self.model_results:
            # Assuming average transaction amount and fraud rates
            avg_transaction = 150  # dollars
            daily_transactions = 1000
            fraud_rate = 0.002  # 0.2%
            
            daily_fraud_loss_prevented = daily_transactions * fraud_rate * avg_transaction * best_fraud_auc
            annual_savings = daily_fraud_loss_prevented * 365
            
            print(f"   Estimated annual fraud loss prevention: ${annual_savings:,.0f}")
        
        return insights
    
    def run_complete_modeling(self):
        """Run complete ML modeling pipeline"""
        print("🚀 Starting Complete ML Modeling Pipeline...")
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare datasets
        print("\n" + "="*80)
        X_fraud, y_fraud = self.prepare_fraud_detection_data()
        X_credit, y_credit = self.prepare_credit_risk_data()
        
        # Train models
        print("\n" + "="*80)
        fraud_results = self.train_fraud_detection_models(X_fraud, y_fraud)
        
        print("\n" + "="*80)
        credit_results = self.train_credit_risk_models(X_credit, y_credit)
        
        # Analyze feature importance
        print("\n" + "="*80)
        self.analyze_feature_importance()
        
        # Generate insights
        print("\n" + "="*80)
        self.generate_model_insights()
        
        # Save everything
        print("\n" + "="*80)
        self.save_models_and_results()
        
        print("\n" + "="*80)
        print("🎉 ML MODELING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📋 Training Summary:")
        
        if 'fraud_detection' in self.model_results:
            fraud_best = self.model_results['fraud_detection']['best_model']
            fraud_auc = self.model_results['fraud_detection']['results'][fraud_best]['auc']
            print(f"   🔍 Fraud Detection: {fraud_best} (AUC: {fraud_auc:.3f})")
        
        if 'credit_risk' in self.model_results:
            credit_best = self.model_results['credit_risk']['best_model']
            credit_auc = self.model_results['credit_risk']['results'][credit_best]['auc']
            print(f"   🏦 Credit Risk: {credit_best} (AUC: {credit_auc:.3f})")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Run financial_risk_dashboards.py for model visualization")
        print(f"   2. Deploy models using api/financial_risk_api.py")
        print(f"   3. Monitor model performance in production")
        print("="*80)
        
        return True

def main():
    """Main function to run ML modeling"""
    modeler = FinancialRiskModels()
    success = modeler.run_complete_modeling()
    
    if success:
        print("✅ ML model training completed successfully!")
    else:
        print("❌ ML modeling failed. Please check data files.")
        
    return success

if __name__ == "__main__":
    main()