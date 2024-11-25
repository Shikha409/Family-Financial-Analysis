import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class FinancialAnalyzer:
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare the data"""
        # Convert Transaction Date to datetime
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        
        # Basic data info
        st.sidebar.header("📊 Data Overview")
        st.sidebar.write(f"Total Records: {len(self.df)}")
        st.sidebar.write(f"Unique Families: {self.df['Family ID'].nunique()}")
        st.sidebar.write(f"Date Range: {self.df['Transaction Date'].min().date()} to {self.df['Transaction Date'].max().date()}")
        
        # Group spending by categories
        self.category_spending = self.df.groupby(['Family ID', 'Category'])['Amount'].sum().reset_index()
        
        # Family level metrics
        self.family_metrics = self.df.groupby('Family ID').agg({
            'Income': 'first',
            'Savings': 'first',
            'Monthly Expenses': 'first',
            'Loan Payments': 'first',
            'Credit Card Spending': 'first',
            'Financial Goals Met (%)': 'first'
        }).reset_index()
    
    def plot_interactive_spending(self, family_id):
        """Create interactive plots"""
        family_data = self.df[self.df['Family ID'] == family_id]
        
        # Category-wise Spending Pie Chart
        category_fig = px.pie(
            family_data, 
            values='Amount', 
            names='Category',
            title=f'Spending Distribution - {family_id}'
        )
        
        # Member-wise Spending Bar Chart
        member_spending = family_data.groupby('Member ID')['Amount'].sum().reset_index()
        member_fig = px.bar(
            member_spending,
            x='Member ID',
            y='Amount',
            title=f'Spending by Family Member - {family_id}'
        )
        
        # Time Series of Spending
        time_series = family_data.groupby('Transaction Date')['Amount'].sum().reset_index()
        time_fig = px.line(
            time_series,
            x='Transaction Date',
            y='Amount',
            title=f'Daily Spending Trend - {family_id}'
        )
        
        return category_fig, member_fig, time_fig
    
    def calculate_family_score(self, family_id):
        """Calculate financial health score"""
        family = self.family_metrics[self.family_metrics['Family ID'] == family_id].iloc[0]
        
        # Financial Ratios Calculation
        savings_ratio = (family['Savings'] / family['Income']) * 100
        expense_ratio = (family['Monthly Expenses'] / family['Income']) * 100
        loan_ratio = (family['Loan Payments'] / family['Income']) * 100
        credit_ratio = (family['Credit Card Spending'] / family['Income']) * 100
        goals_met = family['Financial Goals Met (%)']
        
        # Scoring Weights
        weights = {
            'savings': 0.25,
            'expenses': 0.20,
            'loans': 0.20,
            'credit': 0.15,
            'goals': 0.20
        }
        
        # Calculate Component Scores
        scores = {
            'savings': min(100, savings_ratio * 2) * weights['savings'],
            'expenses': max(0, 100 - expense_ratio) * weights['expenses'],
            'loans': max(0, 100 - (loan_ratio * 2)) * weights['loans'],
            'credit': max(0, 100 - (credit_ratio * 2)) * weights['credit'],
            'goals': goals_met * weights['goals']
        }
        
        # Financial Health Score Visualization
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
        """Generate financial insights"""
        insights = []
        if scores['savings'] < 15:
            insights.append("🔴 Low Savings: Need to increase savings rate")
        if scores['expenses'] < 12:
            insights.append("🔴 High Expenses: Reduce unnecessary spending")
        if scores['loans'] < 12:
            insights.append("🔴 High Loan Burden: Consider debt management")
        if scores['credit'] < 9:
            insights.append("🔴 High Credit Usage: Control credit spending")
        
        recommendations = [
            "💡 Create a strict monthly budget",
            "💡 Build an emergency fund",
            "💡 Explore income growth opportunities",
            "💡 Review and reduce unnecessary expenses"
        ]
        
        return {
            'warnings': insights,
            'recommendations': recommendations
        }

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Family Financial Analysis", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Main Title
    st.title("💰 Family Financial Health Analyzer")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Excel File", 
        type=['xlsx'], 
        help="Upload your family financial transactions Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Read Excel File
            df = pd.read_excel(uploaded_file)
            
            # Create Analyzer
            analyzer = FinancialAnalyzer(df)
            
            # Family Selection
            st.sidebar.header("🏠 Select Family")
            family_ids = df['Family ID'].unique()
            selected_family = st.sidebar.selectbox(
                "Choose Family ID", 
                options=family_ids
            )
            
            # Score and Insights
            score_data = analyzer.calculate_family_score(selected_family)
            
            # Main Analysis Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Financial Health Score")
                st.plotly_chart(score_data['score_visualization'], use_container_width=True)
            
            with col2:
                st.subheader("🚨 Financial Warnings")
                for warning in score_data['insights']['warnings']:
                    st.warning(warning)
                
                st.subheader("💡 Recommendations")
                for rec in score_data['insights']['recommendations']:
                    st.info(rec)
            
            # Detailed Visualizations
            st.header("Detailed Financial Analysis")
            category_plot, member_plot, time_plot = analyzer.plot_interactive_spending(selected_family)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Spending by Category")
                st.plotly_chart(category_plot, use_container_width=True)
                
                st.subheader("Spending by Member")
                st.plotly_chart(member_plot, use_container_width=True)
            
            with col4:
                st.subheader("Daily Spending Trend")
                st.plotly_chart(time_plot, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class FinancialAnalyzer:
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare the data"""
        # Convert Transaction Date to datetime
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        
        # Basic data info
        st.sidebar.header("📊 Data Overview")
        st.sidebar.write(f"Total Records: {len(self.df)}")
        st.sidebar.write(f"Unique Families: {self.df['Family ID'].nunique()}")
        st.sidebar.write(f"Date Range: {self.df['Transaction Date'].min().date()} to {self.df['Transaction Date'].max().date()}")
        
        # Group spending by categories
        self.category_spending = self.df.groupby(['Family ID', 'Category'])['Amount'].sum().reset_index()
        
        # Family level metrics
        self.family_metrics = self.df.groupby('Family ID').agg({
            'Income': 'first',
            'Savings': 'first',
            'Monthly Expenses': 'first',
            'Loan Payments': 'first',
            'Credit Card Spending': 'first',
            'Financial Goals Met (%)': 'first'
        }).reset_index()
    
    def plot_interactive_spending(self, family_id):
        """Create interactive plots"""
        family_data = self.df[self.df['Family ID'] == family_id]
        
        # Category-wise Spending Pie Chart
        category_fig = px.pie(
            family_data, 
            values='Amount', 
            names='Category',
            title=f'Spending Distribution - {family_id}'
        )
        
        # Member-wise Spending Bar Chart
        member_spending = family_data.groupby('Member ID')['Amount'].sum().reset_index()
        member_fig = px.bar(
            member_spending,
            x='Member ID',
            y='Amount',
            title=f'Spending by Family Member - {family_id}'
        )
        
        # Time Series of Spending
        time_series = family_data.groupby('Transaction Date')['Amount'].sum().reset_index()
        time_fig = px.line(
            time_series,
            x='Transaction Date',
            y='Amount',
            title=f'Daily Spending Trend - {family_id}'
        )
        
        return category_fig, member_fig, time_fig
    
    def calculate_family_score(self, family_id):
        """Calculate financial health score"""
        family = self.family_metrics[self.family_metrics['Family ID'] == family_id].iloc[0]
        
        # Financial Ratios Calculation
        savings_ratio = (family['Savings'] / family['Income']) * 100
        expense_ratio = (family['Monthly Expenses'] / family['Income']) * 100
        loan_ratio = (family['Loan Payments'] / family['Income']) * 100
        credit_ratio = (family['Credit Card Spending'] / family['Income']) * 100
        goals_met = family['Financial Goals Met (%)']
        
        # Scoring Weights
        weights = {
            'savings': 0.25,
            'expenses': 0.20,
            'loans': 0.20,
            'credit': 0.15,
            'goals': 0.20
        }
        
        # Calculate Component Scores
        scores = {
            'savings': min(100, savings_ratio * 2) * weights['savings'],
            'expenses': max(0, 100 - expense_ratio) * weights['expenses'],
            'loans': max(0, 100 - (loan_ratio * 2)) * weights['loans'],
            'credit': max(0, 100 - (credit_ratio * 2)) * weights['credit'],
            'goals': goals_met * weights['goals']
        }
        
        # Financial Health Score Visualization
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
        """Generate financial insights"""
        insights = []
        if scores['savings'] < 15:
            insights.append("🔴 Low Savings: Need to increase savings rate")
        if scores['expenses'] < 12:
            insights.append("🔴 High Expenses: Reduce unnecessary spending")
        if scores['loans'] < 12:
            insights.append("🔴 High Loan Burden: Consider debt management")
        if scores['credit'] < 9:
            insights.append("🔴 High Credit Usage: Control credit spending")
        
        recommendations = [
            "💡 Create a strict monthly budget",
            "💡 Build an emergency fund",
            "💡 Explore income growth opportunities",
            "💡 Review and reduce unnecessary expenses"
        ]
        
        return {
            'warnings': insights,
            'recommendations': recommendations
        }

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Family Financial Analysis", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Main Title
    st.title("💰 Family Financial Health Analyzer")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Excel File", 
        type=['xlsx'], 
        help="Upload your family financial transactions Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Read Excel File
            df = pd.read_excel(uploaded_file)
            
            # Create Analyzer
            analyzer = FinancialAnalyzer(df)
            
            # Family Selection
            st.sidebar.header("🏠 Select Family")
            family_ids = df['Family ID'].unique()
            selected_family = st.sidebar.selectbox(
                "Choose Family ID", 
                options=family_ids
            )
            
            # Score and Insights
            score_data = analyzer.calculate_family_score(selected_family)
            
            # Main Analysis Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Financial Health Score")
                st.plotly_chart(score_data['score_visualization'], use_container_width=True)
            
            with col2:
                st.subheader("🚨 Financial Warnings")
                for warning in score_data['insights']['warnings']:
                    st.warning(warning)
                
                st.subheader("💡 Recommendations")
                for rec in score_data['insights']['recommendations']:
                    st.info(rec)
            
            # Detailed Visualizations
            st.header("Detailed Financial Analysis")
            category_plot, member_plot, time_plot = analyzer.plot_interactive_spending(selected_family)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Spending by Category")
                st.plotly_chart(category_plot, use_container_width=True)
                
                st.subheader("Spending by Member")
                st.plotly_chart(member_plot, use_container_width=True)
            
            with col4:
                st.subheader("Daily Spending Trend")
                st.plotly_chart(time_plot, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
