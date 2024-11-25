import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="📊💰 Family Financial Health Analyzer", layout="wide")

class FinancialAnalyzer:
    def __init__(self, excel_data):
        self.df = pd.read_excel(excel_data)
        self.prepare_data()


    # Basic data info
        st.sidebar.header("📊 Data Overview")
        st.sidebar.write(f"Total Records: {len(self.df)}")
        st.sidebar.write(f"Unique Families: {self.df['Family ID'].nunique()}")
        st.sidebar.write(f"Date Range: {self.df['Transaction Date'].min().date()} to {self.df['Transaction Date'].max().date()}")
        
    
    def prepare_data(self):
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        
        # Correlation Matrix
        numeric_columns = ['Income', 'Amount', 'Savings', 'Monthly Expenses', 
                           'Loan Payments', 'Credit Card Spending']
        self.correlation_matrix = self.df[numeric_columns].corr()
        
        # Feature Engineering
        self.df['Expense_to_Income_Ratio'] = self.df['Monthly Expenses'] / self.df['Income'] * 100
        self.df['Savings_Rate'] = self.df['Savings'] / self.df['Income'] * 100
        
        # Existing data preparation
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
        
        # Income Stability Calculation
        family_data = self.df[self.df['Family ID'] == family_id]
        income_std = family_data['Income'].std()
        income_mean = family_data['Income'].mean()
        income_stability = max(0, 100 - (income_std / income_mean * 100)) if income_mean > 0 else 0
        
        savings_ratio = (family['Savings'] / family['Income']) * 100
        expense_ratio = (family['Monthly Expenses'] / family['Income']) * 100
        loan_ratio = (family['Loan Payments'] / family['Income']) * 100
        credit_ratio = (family['Credit Card Spending'] / family['Income']) * 100
        goals_met = family['Financial Goals Met (%)']
        
        weights = {
            'income_stability': 0.2,
            'savings': 0.2,
            'expenses': 0.2,
            'loans': 0.15,
            'credit': 0.15,
            'goals': 0.10
        }
        
        scores = {
            'income_stability': income_stability * weights['income_stability'],
            'savings': min(100, savings_ratio * 2) * weights['savings'],
            'expenses': max(0, 100 - expense_ratio) * weights['expenses'],
            'loans': max(0, 100 - (loan_ratio * 2)) * weights['loans'],
            'credit': max(0, 100 - (credit_ratio * 2)) * weights['credit'],
            'goals': goals_met * weights['goals']
        }
        
        total_score = sum(scores.values())
        
        score_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = total_score,
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
            'total_score': round(total_score, 2),
            'component_scores': scores,
            'score_visualization': score_fig,
            'insights': self.generate_advanced_insights(scores)
        }
    
    def predict_next_month_expenses(self, family_id):
        """Machine Learning based expense prediction"""
        family_data = self.df[self.df['Family ID'] == family_id].copy()
        
        # Feature preparation
        features = ['Income', 'Monthly Expenses', 'Savings', 'Loan Payments']
        X = family_data[features]
        y = family_data['Monthly Expenses']
        
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
    
    def generate_advanced_insights(self, scores):
        insights = {
            'warnings': [],
            'recommendations': []
        }
        
        # Income Stability Insights
        if scores['income_stability'] < 50:
            insights['warnings'].append("🚨 Inconsistent Income: Consider diversifying income sources")
        
        # Savings Rate Insights
        if scores['savings'] < 30:
            insights['warnings'].append("💰 Low Savings Rate: Increase monthly savings")
            insights['recommendations'].append("Set up automatic monthly transfers to savings")
        
        # Expense Management Insights
        if scores['expenses'] < 50:
            insights['warnings'].append("💸 High Expenses: Need to optimize spending")
            insights['recommendations'].append("Create a detailed budget tracking system")
        
        # Debt Health Insights
        if scores['loans'] < 60:
            insights['warnings'].append("🔗 High Debt Burden: Focus on debt reduction")
            insights['recommendations'].append("Explore debt consolidation strategies")
        
        # Goal Progress Insights
        if scores['goals'] < 50:
            insights['warnings'].append("🎯 Low Goal Achievement: Revise financial goals")
            insights['recommendations'].append("Break down long-term goals into smaller milestones")
        
        return insights

# Main Application
def main():
    st.title("📊💰  Family Financial Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            analyzer = FinancialAnalyzer(uploaded_file)
            
            # Sidebar for family selection
            st.sidebar.title("Select Family")
            family_ids = analyzer.df['Family ID'].unique()
            family_id = st.sidebar.selectbox(
                "Choose Family ID",
                options=family_ids
            )
            
            # Correlation Heatmap
            st.subheader("Financial Features Correlation")
            correlation_fig = px.imshow(
                analyzer.correlation_matrix, 
                text_auto=True, 
                title="Correlation Matrix of Financial Indicators"
            )
            st.plotly_chart(correlation_fig, use_container_width=True)
            
            # Calculate scores and get insights
            score_data = analyzer.calculate_family_score(family_id)
            
            # Visualize Scoring Components
            component_fig = go.Figure(data=[
                go.Bar(
                    x=list(score_data['component_scores'].keys()),
                    y=list(score_data['component_scores'].values()),
                    marker_color=['blue', 'green', 'red', 'purple', 'orange', 'cyan']
                )
            ])
            component_fig.update_layout(title="Financial Health Component Scores")
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(score_data['score_visualization'], use_container_width=True)
                st.plotly_chart(component_fig, use_container_width=True)
            
            with col2:
                st.subheader("Insights")
                for warning in score_data['insights']['warnings']:
                    st.warning(warning)
                
                st.subheader("Recommendations")
                for rec in score_data['insights']['recommendations']:
                    st.info(rec)
            
            # Expense Prediction
            st.subheader("Next Month Expense Prediction")
            prediction = analyzer.predict_next_month_expenses(family_id)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric(
                    "Predicted Next Month Expense", 
                    f"₹{prediction['predicted_expense']:,.2f}"
                )
            
            with col4:
                st.metric(
                    "Prediction Model Accuracy", 
                    f"{prediction['model_accuracy']}%"
                )
            
            # Get spending visualizations
            category_plot, member_plot, time_plot = analyzer.plot_interactive_spending(family_id)
            
            # Display plots in columns
            st.subheader("Detailed Spending Analysis")
            col5, col6 = st.columns(2)
            
            with col5:
                st.plotly_chart(category_plot, use_container_width=True)
                st.plotly_chart(member_plot, use_container_width=True)
            
            with col6:
                st.plotly_chart(time_plot, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
    else:
        st.info("Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
