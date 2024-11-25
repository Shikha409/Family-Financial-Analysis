import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Family Financial Analysis", layout="wide")

# Main title
st.title("📊 Family Financial Analysis Dashboard")

# FinancialAnalyzer Class
class FinancialAnalyzer:
    def __init__(self, excel_data):
        self.df = pd.read_excel(excel_data)
        self.prepare_data()
    
    def prepare_data(self):
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        self.category_spending = self.df.groupby(['Family ID', 'Category'])['Amount'].sum().reset_index()
        self.family_metrics = self.df.groupby('Family ID').agg({
            'Income': 'first',
            'Savings': 'first',
            'Monthly Expenses': 'first',
            'Loan Payments': 'first',
            'Credit Card Spending': 'first',
            'Financial Goals Met (%)': 'first'
        }).reset_index()
    
    def plot_interactive_spending(self, family_id):
        family_data = self.df[self.df['Family ID'] == family_id]
        
        category_fig = px.pie(
            family_data, 
            values='Amount', 
            names='Category',
            title=f'Category-wise Spending - {family_id}'
        )
        
        member_spending = family_data.groupby('Member ID')['Amount'].sum().reset_index()
        member_fig = px.bar(
            member_spending,
            x='Member ID',
            y='Amount',
            title=f'Member-wise Spending - {family_id}'
        )
        
        time_series = family_data.groupby('Transaction Date')['Amount'].sum().reset_index()
        time_fig = px.line(
            time_series,
            x='Transaction Date',
            y='Amount',
            title=f'Daily Spending Trend - {family_id}'
        )
        
        return category_fig, member_fig, time_fig
    
    def calculate_family_score(self, family_id):
        family = self.family_metrics[self.family_metrics['Family ID'] == family_id].iloc[0]
        
        savings_ratio = (family['Savings'] / family['Income']) * 100
        expense_ratio = (family['Monthly Expenses'] / family['Income']) * 100
        loan_ratio = (family['Loan Payments'] / family['Income']) * 100
        credit_ratio = (family['Credit Card Spending'] / family['Income']) * 100
        goals_met = family['Financial Goals Met (%)']
        
        weights = {
            'savings': 0.25,
            'expenses': 0.20,
            'loans': 0.20,
            'credit': 0.15,
            'goals': 0.20
        }
        
        scores = {
            'savings': min(100, savings_ratio * 2) * weights['savings'],
            'expenses': max(0, 100 - expense_ratio) * weights['expenses'],
            'loans': max(0, 100 - (loan_ratio * 2)) * weights['loans'],
            'credit': max(0, 100 - (credit_ratio * 2)) * weights['credit'],
            'goals': goals_met * weights['goals']
        }
        
        score_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sum(scores.values()),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Financial Health Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ]
            }
        ))
        
        return {
            'total_score': round(sum(scores.values()), 2),
            'component_scores': scores,
            'score_visualization': score_fig,
            'insights': self.generate_insights(scores)
        }
    
    def generate_insights(self, scores):
        insights = []
        if scores['savings'] < 15:
            insights.append("🔴 Savings are below recommended levels")
        if scores['expenses'] < 12:
            insights.append("🔴 Monthly expenses are high")
        if scores['loans'] < 12:
            insights.append("🔴 Loan payments are high")
        if scores['credit'] < 9:
            insights.append("🔴 High credit card usage")
        if scores['goals'] < 12:
            insights.append("🔴 Financial goals need attention")
        
        recommendations = [
            "💡 Reduce non-essential expenses",
            "💡 Consider debt consolidation",
            "💡 Create monthly budget",
            "💡 Set specific financial goals"
        ]
        
        return {
            'warnings': insights,
            'recommendations': recommendations
        }

# AdvancedFinancialAnalyzer Class
class AdvancedFinancialAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        numeric_columns = ['Income', 'Amount', 'Savings', 'Monthly Expenses', 
                           'Loan Payments', 'Credit Card Spending']
        self.correlation_matrix = self.df[numeric_columns].corr()
        self.df['Expense_to_Income_Ratio'] = self.df['Monthly Expenses'] / self.df['Income'] * 100
        self.df['Savings_Rate'] = self.df['Savings'] / self.df['Income'] * 100
    
    def advanced_financial_scoring(self, family_id):
        family_data = self.df[self.df['Family ID'] == family_id]
        # Similar logic as provided by you for score calculations
        
        return {"dummy": "data"} # Replace with actual code

    def predict_next_month_expenses(self, family_id):
        return {"dummy_prediction": "data"} # Replace with actual logic

# Main App Logic
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])

if uploaded_file:
    st.sidebar.title("Select Analysis")
    analysis_type = st.sidebar.radio("Choose Analysis Type", ["Basic", "Advanced"])

    if analysis_type == "Basic":
        analyzer = FinancialAnalyzer(uploaded_file)
        # Implement basic functionality UI

    elif analysis_type == "Advanced":
        df = pd.read_excel(uploaded_file)
        adv_analyzer = AdvancedFinancialAnalyzer(df)
        # Implement advanced UI features
