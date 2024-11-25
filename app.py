import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class AdvancedFinancialAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        """Advanced data preparation with correlation analysis"""
        try:
            # Debug Log
            st.info("Preparing data for analysis...")

            # Ensure date parsing
            self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'], errors='coerce')
            st.success("Transaction Date parsing complete.")
            
            # Correlation Matrix
            numeric_columns = ['Income', 'Amount', 'Savings', 'Monthly Expenses', 
                               'Loan Payments', 'Credit Card Spending']
            
            missing_cols = [col for col in numeric_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in dataset: {missing_cols}")

            correlation_matrix = self.df[numeric_columns].corr()
            
            # Feature Engineering
            self.df['Expense_to_Income_Ratio'] = (self.df['Monthly Expenses'] / 
                                                  self.df['Income']).fillna(0) * 100
            self.df['Savings_Rate'] = (self.df['Savings'] / 
                                       self.df['Income']).fillna(0) * 100
            
            # Store correlation for later use
            self.correlation_matrix = correlation_matrix
        except Exception as e:
            st.error(f"Error during data preparation: {e}")
            raise
    
    def advanced_financial_scoring(self, family_id):
        """Enhanced financial scoring model"""
        try:
            family_data = self.df[self.df['Family ID'] == family_id]
            if family_data.empty:
                raise ValueError(f"No data found for Family ID: {family_id}")
            
            # Advanced Scoring Components
            scoring_components = {
                'income_stability': self._calculate_income_stability(family_data),
                'savings_rate': self._calculate_savings_rate(family_data),
                'expense_management': self._calculate_expense_management(family_data),
                'debt_health': self._calculate_debt_health(family_data),
                'financial_goal_progress': self._calculate_goal_progress(family_data)
            }
            
            # Weighted Scoring
            weights = {
                'income_stability': 0.2,
                'savings_rate': 0.25,
                'expense_management': 0.2,
                'debt_health': 0.2,
                'financial_goal_progress': 0.15
            }
            
            # Calculate Composite Score
            total_score = sum(
                score * weights.get(component, 0) 
                for component, score in scoring_components.items()
            )
            
            return {
                'total_score': total_score,
                'component_scores': scoring_components,
                'insights': self._generate_advanced_insights(scoring_components)
            }
        except Exception as e:
            st.error(f"Error during financial scoring: {e}")
            raise
    
    def _calculate_income_stability(self, family_data):
        """Calculate income stability score"""
        try:
            income_std = family_data['Income'].std()
            income_mean = family_data['Income'].mean()
            stability_score = max(0, 100 - (income_std / income_mean * 100))
            return round(stability_score, 2)
        except ZeroDivisionError:
            return 0
    
    def _calculate_savings_rate(self, family_data):
        """Calculate savings rate score"""
        try:
            savings_rate = (family_data['Savings'] / family_data['Income']).mean() * 100
            return round(min(100, savings_rate * 2), 2)
        except ZeroDivisionError:
            return 0
    
    def _calculate_expense_management(self, family_data):
        """Calculate expense management score"""
        try:
            expense_to_income = (family_data['Monthly Expenses'] / family_data['Income']).mean() * 100
            return round(max(0, 100 - expense_to_income), 2)
        except ZeroDivisionError:
            return 0
    
    def _calculate_debt_health(self, family_data):
        """Calculate debt health score"""
        try:
            total_debt = family_data['Loan Payments'].mean()
            income = family_data['Income'].mean()
            debt_ratio = (total_debt / income) * 100
            return round(max(0, 100 - (debt_ratio * 2)), 2)
        except ZeroDivisionError:
            return 0
    
    def _calculate_goal_progress(self, family_data):
        """Calculate financial goal progress"""
        try:
            goal_met_percentage = family_data['Financial Goals Met (%)'].mean()
            return round(goal_met_percentage, 2)
        except KeyError:
            return 0
    
    def _generate_advanced_insights(self, scores):
        """Generate advanced financial insights"""
        insights = {
            'warnings': [],
            'recommendations': []
        }
        
        # Income Stability Insights
        if scores['income_stability'] < 50:
            insights['warnings'].append("🚨 Inconsistent Income: Consider diversifying income sources")
        
        # Savings Rate Insights
        if scores['savings_rate'] < 30:
            insights['warnings'].append("💰 Low Savings Rate: Increase monthly savings")
            insights['recommendations'].append("Set up automatic monthly transfers to savings")
        
        # Expense Management Insights
        if scores['expense_management'] < 50:
            insights['warnings'].append("💸 High Expenses: Need to optimize spending")
            insights['recommendations'].append("Create a detailed budget tracking system")
        
        # Debt Health Insights
        if scores['debt_health'] < 60:
            insights['warnings'].append("🔗 High Debt Burden: Focus on debt reduction")
            insights['recommendations'].append("Explore debt consolidation strategies")
        
        # Goal Progress Insights
        if scores['financial_goal_progress'] < 50:
            insights['warnings'].append("🎯 Low Goal Achievement: Revise financial goals")
            insights['recommendations'].append("Break down long-term goals into smaller milestones")
        
        return insights
    
    def predict_next_month_expenses(self, family_id):
        """Machine Learning based expense prediction"""
        try:
            family_data = self.df[self.df['Family ID'] == family_id].copy()
            
            # Feature preparation
            features = ['Income', 'Monthly Expenses', 'Savings', 'Loan Payments']
            X = family_data[features]
            y = family_data['Monthly Expenses']
            
            if X.empty or y.empty:
                raise ValueError("Insufficient data for prediction")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Random Forest Regression
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            
            # Predict next month's expenses
            last_data_point = scaler.transform(X.iloc[-1:])
            predicted_expense = rf_model.predict(last_data_point)[0]
            
            return {
                'predicted_expense': round(predicted_expense, 2),
                'model_accuracy': round(rf_model.score(X_test_scaled, y_test) * 100, 2)
            }
        except Exception as e:
            st.error(f"Error during expense prediction: {e}")
            raise
