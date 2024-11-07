import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import cvxpy as cp
import scipy.optimize as sco
from pypfopt import EfficientFrontier, expected_returns, risk_models, CLA
import time
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import hashlib
# -------------------------------
# 1. Imports and Data Loading
# -------------------------------

# # Load the returns data
# df = pd.read_excel(
#     r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\DS_RI_T_USD_M.xlsx",
#     header=None,
# )

# # Transpose the DataFrame
# df = df.T

# # Set the second row (index 1) as the column headers
# df.columns = df.iloc[0]
# column_names = df.iloc[1].values
# print(column_names)

# # Remove the first two rows as they are now redundant
# df = df.drop([0, 1])

# # Rename the first column to 'Date' and set it as the index
# df = df.rename(columns={df.columns[0]: "Date"}).set_index("Date")

# # Convert all entries to floats for uniformity and handling
# df = df.astype(float)

# # Initialize a set to keep track of dropped stocks
# dropped_stocks = set()

# # 1. Remove stocks with initial zero prices
# initial_zeros = df.iloc[0] == 0
# dropped_stocks.update(df.columns[initial_zeros])
# print(f"Initial zero : {df.columns[initial_zeros]}")
# df = df.loc[:, ~initial_zeros]

# # 2. Remove stocks that ever drop to zero
# ever_zeros = (df == 0).any()
# dropped_stocks.update(df.columns[ever_zeros])
# print(f"Ever zero : {df.columns[ever_zeros]}")
# df = df.loc[:, ~ever_zeros]

# # 3. Remove stocks that do not recover after dropping to zero
# max_prior = df.cummax()
# recovered = ((df / max_prior.shift()) > 0.1).any()
# non_recovered = df.columns[~recovered]
# dropped_stocks.update(non_recovered)
# print(f"Non recovered : {non_recovered}")
# df = df.loc[:, recovered]

# # # Filter based on sector information
# # static_file = pd.read_excel(
# #     r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\Static.xlsx"
# # )
# # sectors = ["Energy", "Materials", "Utilities", "Industrials"]
# # companies = static_file[static_file["GICSSectorName"].isin(sectors)]
# # isin_list = companies["ISIN"].tolist()

# # # Identify stocks that are not in the highly polluting sectors
# # non_polluting_stocks = set(df.columns) - set(isin_list)
# # dropped_stocks.update(non_polluting_stocks)

# # df = df[df.columns.intersection(isin_list)]


# # # Reset column names to the original names after modifications
# # df.columns = column_names[
# #     1 : len(df.columns) + 1
# # ]  # Skip the first name since it corresponds to the Date column

# # Proceed with any further data processing, such as calculating returns
# monthly_returns = df.pct_change()
# monthly_returns = monthly_returns.drop(monthly_returns.index[0])

# # Handling NaN and infinite values
# monthly_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
# monthly_returns.interpolate(method="linear", axis=0, inplace=True)
# monthly_returns.fillna(method="ffill", axis=0, inplace=True)
# monthly_returns.fillna(method="bfill", axis=0, inplace=True)

# # Display results
# print("Remaining NaN values in monthly returns:", monthly_returns.isnull().sum().sum())
# df.to_csv("Cleaned_df.csv", index=True)
# monthly_returns.to_csv("Cleaned_df_returns.csv", index=True)


data = pd.read_csv("Cleaned_df.csv", index_col="Date")
static_data = pd.read_excel(
    r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\Static.xlsx"
)
print(data.head())
# data = data[
#     [
#         "AN8068571086",
#         "ARALUA010258",
#         "ARP125991090",
#         "ARSIDE010029",
#         "AT00000VIE62",
#         "AT0000652011",
#         "AT0000743059",
#         "AT0000746409",
#         "AT0000831706",
#         "AT0000908504",
#         "ZAE000109815",
#         "ZAE000117321",
#         "ZAE000134961",
#         "ZAE000170049",
#         "ZAE000179420",
#         "ZAE000191342",
#         "ZAE000255915",
#         "ZAE000298253",
#         "ZAE000302618",
#         "ZAE000322095",
#     ]
# ]
assets = data.columns.tolist()


# -------------------------------
# 2. Risk Aversion Quiz
# -------------------------------


# Risk aversion quiz using a form
# Risk assessment questions
questions = [
    {"type": "text", "question": "What is your name?", "key": "name"},
    {"type": "initial", "key": "initial", "questions": [
        {"type": "number", "question": "Age", "key": "age"},
        {"type": "select", "question": "Marital status", "key": "marital_status",
         "options": ["Single", "Common law", "Married", "Separated", "Divorced", "Widowed"]},
        {"type": "number", "question": "How many dependents do you have?", "min_value": 0, "max_value": 10, "key": "dependents"}
    ]},
    {"type": "employment_income", "key": "employment_income", "questions": [
        {"type": "radio", "question": "Are you currently employed?", "key": "employed",
         "options": ["Yes", "No"]},
        {"type": "number", "question": "What is your annual household income?", "key": "income"},
        {"type": "buttons", "question": "Which statement best describes your home ownership status?", "key": "home_ownership",
         "options": ["I don't own a home", "I'm paying a mortgage", "My mortgage is paid off"]}
    ]},
    {"type": "assets_liabilities", "key": "assets_liabilities", "questions": [
        {"type": "number", "question": "What is the total value of all your assets?", "key": "total_assets"},
        {"type": "number", "question": "What is the value of your fixed assets (e.g., property, vehicles)?", "key": "fixed_assets"},
        {"type": "number", "question": "What is the total value of your liabilities?", "key": "liabilities"}
    ]},
    {"type": "multiselect", "question": "What are your primary financial goals?", 
     "options": ["Retirement", "Home purchase", "Education", "Emergency fund", "Wealth accumulation"], 
     "key": "financial_goals"},
    {"type": "select", "question": "Which life stage best describes you?", 
     "options": ["Starting out", "Career building", "Peak earning years", "Pre-retirement", "Retirement"], 
     "key": "life_stage"},
    {"type": "image_buttons", "question": "How would you describe your investment experience?", 
     "options": [
         {"text": "Mostly Cash Savings", "image": "💰", "key": "cash_savings"},
         {"text": "Bonds, Income funds, GICs", "image": "📊", "key": "bonds_income"},
         {"text": "Mutual Funds and Exchange Traded Funds (ETFs)", "image": "📈", "key": "mutual_etfs"},
         {"text": "Self-Directed Investor: Stocks, Equities, Cryptocurrencies", "image": "🚀", "key": "self_directed"}
     ], 
     "key": "investment_experience"},
    {"type": "radio", "question": "How would you react if your investment lost 20% in a year?", 
     "options": ["Sell all investments", "Sell some", "Hold steady", "Buy more", "Buy a lot more"], "key": "market_reaction"},
    {"type": "chart", "question": "What level of volatility would you be the most comfortable with?", 
     "options": ["Low Volatility", "Balanced", "High Volatility"], 
     "key": "volatility_preference"},
    {"type": "radio", "question": "How long do you plan to hold your investments?", 
     "options": ["0-3 years", "3-5 years", "5+ years"], "key": "investment_horizon"},
    {"type": "radio", "question": "What's your risk capacity (ability to take risks)?", 
     "options": ["Very low", "Low", "Medium", "High", "Very high"], "key": "risk_capacity"},
    {"type": "slider", "question": "How confident are you in your investment knowledge?", 
     "min_value": 0, "max_value": 10, "step": 1, "key": "investment_confidence"}
]

def create_investment_chart(volatility_level):
    np.random.seed(42)  # For reproducibility
    x = np.arange(100)
    
    if volatility_level == "Low Volatility":
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise) * 0.3
    elif volatility_level == "Balanced":
        trend = np.linspace(0, 20, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise)
    else:  # High Volatility
        trend = np.linspace(0, 40, 100)  # Steeper overall trend
        volatility = np.random.normal(0, 1, 100) * 3  # Increased volatility
        momentum = np.cumsum(np.random.normal(0, 0.1, 100))  # Add momentum
        y = 100 + trend + np.cumsum(volatility) + momentum * 10
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('x', axis=alt.Axis(title='Time')),
        y=alt.Y('y', axis=alt.Axis(title='Value'), scale=alt.Scale(domain=[df.y.min()-10, df.y.max()+10])),
        tooltip=['x', 'y']
    ).properties(
        width=200,
        height=150,
        title=f"{volatility_level}"
    )
    
    return chart

def prepare_data_for_ml(answers):
    data = {}
    for key, value in answers.items():
        question = next((q for q in questions if q.get('key') == key), None)
        if question:
            if key == 'name':
                # Skip the name to maintain anonymity
                continue
            elif question['type'] == 'image_buttons' and key == 'investment_experience':
                # Handle investment experience separately
                for option in question['options']:
                    data[f"{key}_{option['key']}"] = 1 if value == option['text'] else 0
            elif 'options' in question:
                if isinstance(value, list):  # For multiselect questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option in value else 0
                elif isinstance(value, str):  # For categorical questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option == value else 0
            else:  # For numerical or text questions without options
                data[key] = value
        else:
            # Handle nested questions
            for q in questions:
                if q['type'] in ['initial', 'employment_income', 'assets_liabilities']:
                    sub_question = next((sq for sq in q['questions'] if sq.get('key') == key), None)
                    if sub_question:
                        if 'options' in sub_question and isinstance(value, str):
                            for option in sub_question['options']:
                                data[f"{key}_{option}"] = 1 if option == value else 0
                        else:
                            data[key] = value
                        break
    return data
    
def train_ml_model():
    data = pd.read_csv('user_data.csv')
    X = data.drop('risk_score', axis=1)
    y = data['risk_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'risk_assessment_model.joblib')
    return model

def calculate_risk_score_ml(answers):
    MIN_SAMPLES = 30  # Minimum number of samples required for ML

    if os.path.exists('user_data.csv'):
        data = pd.read_csv('user_data.csv')
        if len(data) >= MIN_SAMPLES:
            try:
                model = joblib.load('risk_assessment_model.joblib')
            except:
                model = train_ml_model()
            
            data = prepare_data_for_ml(answers)
            features = pd.DataFrame([data])
            return int(model.predict(features)[0])
        else:
            print(f"Using rule-based model. Need {MIN_SAMPLES - len(data)} more samples for ML.")
    else:
        print("No data file found. Using rule-based model.")

    return calculate_risk_score_rule_based(answers)
    
def calculate_risk_score_rule_based(answers):
    score = 0
    weights = {
        "age": lambda x: max(0, min(10, (65 - x) / 4)),  # Increased impact, 0-10 range
        "marital_status": {"Single": 8, "Common law": 6, "Married": 4, "Separated": 3, "Divorced": 2, "Widowed": 1},
        "dependents": lambda x: max(0, 8 - x * 2),  # 0 dependents = 8, 1 = 6, 2 = 4, 3 = 2, 4+ = 0
        "employed": {"Yes": 5, "No": 0},
        "income": lambda x: min(8, x / 25000),  # 1 point per $25k, max 8 points
        "home_ownership": {"I don't own a home": 0, "I'm paying a mortgage": 4, "My mortgage is paid off": 8},
        "investment_experience": {
            "Mostly Cash Savings and GICs": 0,
            "Bonds, Income funds, GICs": 3,
            "Mutual Funds and Exchange Traded Funds (ETFs)": 6,
            "Self-Directed Investor: Stocks, Equities, Cryptocurrencies": 10
        },
        "market_reaction": {"Sell all investments": 0, "Sell some": 3, "Hold steady": 6, "Buy more": 8, "Buy a lot more": 10},
        "volatility_preference": {"Low Volatility": 0, "Balanced": 5, "High Volatility": 10},
        "investment_horizon": {"0-3 years": 0, "3-5 years": 5, "5+ years": 10},
        "risk_capacity": {"Very low": 0, "Low": 3, "Medium": 6, "High": 8, "Very high": 10}
    }
    
    for key, value in answers.items():
        if key in weights:
            if callable(weights[key]):
                score += weights[key](value)
            elif isinstance(weights[key], dict):
                score += weights[key].get(value, 0)
            elif isinstance(value, (int, float)):
                score += value * weights[key]
    
    # Calculate net worth and add to score
    total_assets = answers.get('total_assets', 0)
    liabilities = answers.get('liabilities', 0)
    net_worth = total_assets - liabilities
    
    # Add net worth factor to score (0-10 points)
    net_worth_score = min(10, max(0, net_worth / 100000))  # 1 point per $100k net worth, max 10 points
    score += net_worth_score
    
    # Add liquidity factor to score (0-5 points)
    liquid_assets = total_assets - answers.get('fixed_assets', 0)
    liquidity_ratio = liquid_assets / total_assets if total_assets > 0 else 0
    liquidity_score = liquidity_ratio * 5  # 0-5 points based on liquidity ratio
    score += liquidity_score
    
    return score  # Note: We're not converting to int here to allow for more granularity

def get_risk_tolerance(score):
    if score < 30:
        return "Conservative"
    elif score < 50:
        return "Moderately Conservative"
    elif score < 70:
        return "Balanced"
    elif score < 90:
        return "Moderately Aggressive"
    else:
        return "Aggressive"
def display_summary(answers, risk_tolerance):
    st.subheader("Assessment Summary")
    for question in questions:
        if question['key'] in answers:
            st.write(f"**{question['question']}** Your answer: {answers[question['key']]}")
    
    st.subheader("Risk Tolerance Explanation")
    st.write(f"Based on your answers, your risk tolerance is: **{risk_tolerance}**")
    st.write("This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility.")

def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS for larger and evenly spaced investment experience buttons
    st.markdown("""
    <style>
    .investment-button {
        width: 100%;
        height: 120px;
        white-space: normal;
        word-wrap: break-word;
        padding: 10px;
        font-size: 14px;
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .investment-button .emoji {
        font-size: 24px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Investment Risk Tolerance Assessment")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Display progress
    st.progress(st.session_state.step / len(questions))

    # Back and Start Over buttons
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("← Back") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    with col3:
        if st.button("↻ Start over"):
            st.session_state.step = 0
            st.session_state.user_answers = {}
            st.session_state.assessment_complete = False
            st.session_state.summary = None
            st.session_state.results = None
            st.rerun()

    # Display current question(s) or results
    if not st.session_state.assessment_complete:
        if st.session_state.step < len(questions):
            q = questions[st.session_state.step]
            
            if q['type'] == 'initial':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'select':
                        st.session_state.user_answers[sub_q['key']] = st.selectbox(sub_q['question'], sub_q['options'], key=sub_q['key'])
                    elif sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], key=sub_q['key'])
            elif q['type'] == 'employment_income':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], horizontal=True, key=sub_q['key'])
                    elif sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'buttons':
                        st.write(sub_q['question'])
                        cols = st.columns(3)
                        for i, option in enumerate(sub_q['options']):
                            if cols[i].button(option, key=f"{sub_q['key']}_{i}"):
                                st.session_state.user_answers[sub_q['key']] = option
            elif q['type'] == 'assets_liabilities':
                st.header("Tell us about your assets and liabilities")
                for sub_q in q['questions']:
                    st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0.0, value=0.0, step=1000.0, key=sub_q['key'])
            elif q['type'] == 'radio':
                st.session_state.user_answers[q['key']] = st.radio(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'chart':
                st.write(q['question'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.altair_chart(create_investment_chart("Low Volatility"))
                with col2:
                    st.altair_chart(create_investment_chart("Balanced"))
                with col3:
                    st.altair_chart(create_investment_chart("High Volatility"))
                st.session_state.user_answers[q['key']] = st.radio("Select your preferred volatility level:", q['options'], key=q['key'])
            elif q['type'] == 'text':
                st.session_state.user_answers[q['key']] = st.text_input(q['question'], key=q['key'])
            elif q['type'] == 'multiselect':
                st.session_state.user_answers[q['key']] = st.multiselect(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'select':
                st.session_state.user_answers[q['key']] = st.selectbox(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'slider':
                st.session_state.user_answers[q['key']] = st.slider(q['question'], q['min_value'], q['max_value'], q['step'], key=q['key'])
            elif q['type'] == 'image_buttons':
                cols = st.columns(len(q['options']))
                for i, option in enumerate(q['options']):
                    if cols[i].button(f"{option['image']} {option['text']}", key=f"{q['key']}_{i}"):
                        st.session_state.user_answers[q['key']] = option['text']
                st.write(f"Selected: {st.session_state.user_answers.get(q['key'], 'None')}")
            
            # Next button with validation
            if st.button("Next"):
                if q['type'] == 'initial' or q['type'] == 'employment_income' or q['type'] == 'assets_liabilities':
                    # Check if all sub-questions are answered
                    all_answered = all(sub_q['key'] in st.session_state.user_answers for sub_q in q['questions'])
                else:
                    # For other question types, check if the main question is answered
                    all_answered = q['key'] in st.session_state.user_answers and st.session_state.user_answers[q['key']]
                
                if all_answered:
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.error("Please answer all questions to continue.")

        # Final submission
        elif st.session_state.step == len(questions):
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Submit", use_container_width=True):
                    risk_score = calculate_risk_score_ml(st.session_state.user_answers)
                    risk_tolerance = get_risk_tolerance(risk_score)

                    st.session_state.results = {
                        'risk_tolerance': risk_tolerance,
                    }

                    user_data = {
                        **st.session_state.user_answers,
                        'risk_score': risk_score,
                        'risk_tolerance': risk_tolerance,
                    }


    # Display results
    if st.session_state.assessment_complete:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        if st.session_state.results:
            st.write(f"Your risk tolerance is: {st.session_state.results['risk_tolerance']}")
            st.write(f"Recommended allocation: {st.session_state.results['equity_allocation']}% Equities, {st.session_state.results['income_allocation']}% Income")
        st.success("Your profile has been saved!")

        # Add disclaimer on the final page
        st.markdown("""
        ---
        **Disclaimer**: This risk assessment tool is for educational purposes only and does not constitute financial advice. 
        Please consult with a qualified financial advisor before making any investment decisions.
        """)

def generate_summary(answers, risk_tolerance):
    summary = "## Assessment Summary\n\n"
    for question in questions:
        if question['key'] in answers:
            summary += f"**{question['question']}** Your answer: {answers[question['key']]}\n\n"
    
    summary += f"## Risk Tolerance Explanation\n\n"
    summary += f"Based on your answers, your risk tolerance is: **{risk_tolerance}**\n\n"
    summary += "This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility."
    
    return summary

# Add this at the end of your script
def close_db_connection():
    global conn
    conn.close()

if __name__ == "__main__":
    main()
    close_db_connection()


# -------------------------------
# 3. Constraints Selection and Parameters
# -------------------------------

# Initialize session state variables
if "optimization_run" not in st.session_state:
    st.session_state["optimization_run"] = False
if "weights" not in st.session_state:
    st.session_state["weights"] = None
if "mean_returns" not in st.session_state:
    st.session_state["mean_returns"] = None
if "cov_matrix" not in st.session_state:
    st.session_state["cov_matrix"] = None
if "previous_params" not in st.session_state:
    st.session_state["previous_params"] = None


# Constraints
st.header("Constraints Selection")
long_only = st.checkbox("Long only", value=True)
use_sentiment = st.checkbox("Use sentiment data?")
sectors_filter = st.checkbox("Sectors filter")
country_filter = st.checkbox("Country filter")
carbon_footprint = st.checkbox("Carbon footprint")
min_weight_constraint = st.checkbox("Minimum weight constraint")
max_weight_constraint = st.checkbox("Maximum weight constraint")
leverage_limit = st.checkbox("Leverage limit")

# Choose objective function
st.header("Choose Optimization Objective")
objectives = [
    "Maximum Sharpe Ratio Portfolio",
    "Minimum Global Variance Portfolio",
    "Maximum Diversification Portflio",
    "Equally Weighted Risk Contribution Portfolio",
    "Inverse Volatility Portfolio",
]
selected_objective = st.multiselect("Select an objective function", objectives)

# Risk-Free Asset Inclusion
st.header("Risk-Free Asset Inclusion")
include_risk_free_asset = st.checkbox(
    "Include a Risk-Free Asset in the Optimization?", value=True
)

if include_risk_free_asset:
    risk_free_rate = st.number_input(
        "Enter the risk-free rate (e.g., 0.01 for 1%)",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
    )
else:
    risk_free_rate = 0.0


# Additional inputs
if sectors_filter:
    sectors = static_data["GICSSectorName"].unique().tolist()
    selected_sectors = st.multiselect("Select sectors to include", sectors)
else:
    selected_sectors = None


if country_filter:
    countries = static_data["Country"].unique().tolist()
    selected_countries = st.multiselect("Select countries to include", countries)
else:
    selected_countries = None

if min_weight_constraint:
    min_weight_value = (
        st.number_input(
            "Minimum weight (%)", min_value=-100.0, max_value=100.0, value=-100.0
        )
        / 100
    )
else:
    min_weight_value = 0.0
if max_weight_constraint:
    max_weight_value = (
        st.number_input(
            "Maximum weight (%)", min_value=0.0, max_value=100.0, value=100.0
        )
        / 100
    )
else:
    max_weight_value = 1.0

if leverage_limit:
    leverage_limit_value = st.number_input("Leverage limit", min_value=0.0, value=1.0)
else:
    leverage_limit_value = 1.0


# Function to get current parameters
def get_current_params():
    params = {
        "long_only": long_only,
        "use_sentiment": use_sentiment,
        "sectors_filter": sectors_filter,
        "selected_sectors": (
            tuple(sorted(selected_sectors)) if selected_sectors else None
        ),
        "country_filter": country_filter,
        "selected_countries": (
            tuple(sorted(selected_countries)) if selected_countries else None
        ),
        "carbon_footprint": carbon_footprint,
        "min_weight_constraint": min_weight_constraint,
        "min_weight_value": min_weight_value,
        "max_weight_constraint": max_weight_constraint,
        "max_weight_value": max_weight_value,
        "leverage_limit": leverage_limit,
        "leverage_limit_value": leverage_limit_value,
        "include_risk_free_asset": include_risk_free_asset,
        "risk_free_rate": risk_free_rate,
        # Include risk_aversion if it can change
        "risk_aversion": risk_aversion,
    }
    return params


# Get current parameters
current_params = get_current_params()
previous_params = st.session_state.get("previous_params", None)

# Compare current and previous parameters
if previous_params is not None and current_params != previous_params:
    st.session_state["optimization_run"] = False

# Update previous parameters
st.session_state["previous_params"] = current_params

# -------------------------------
# 4. Data Filtering Based on Sectors and Countries
# -------------------------------

# Output the total number of stocks before filtering
st.write(f"Total number of stocks before filtering: {data.shape[1]}")


# Filtering based on sectors and countries using ISIN numbers
def filter_stocks(data, sectors=None, countries=None):
    all_isins = data.columns.tolist()

    if sectors is not None:
        companies_sector = static_data[static_data["GICSSectorName"].isin(sectors)]
        sector_isins = companies_sector["ISIN"].tolist()
        all_isins = list(set(all_isins).intersection(set(sector_isins)))
        st.write(f"Total number of stocks after sector filtering: {len(all_isins)}")

    if countries is not None:
        companies_country = static_data[static_data["Country"].isin(countries)]
        country_isins = companies_country["ISIN"].tolist()
        all_isins = list(set(all_isins).intersection(set(country_isins)))
        st.write(f"Total number of stocks after country filtering: {len(all_isins)}")

    data_filtered = data[all_isins]
    return data_filtered


# Apply filtering
data = filter_stocks(data, sectors=selected_sectors, countries=selected_countries)

# Assets list after filtering
assets = data.columns.tolist()

# -------------------------------
# 5. Optimization Function
# -------------------------------


def adjust_covariance_matrix(cov_matrix, delta=1e-5):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    if np.all(eigenvalues <= 0):

        # Adjust negative eigenvalues
        adjusted_eigenvalues = np.where(eigenvalues > delta, eigenvalues, delta)

        # Reconstruct the covariance matrix
        cov_matrix_adjusted = (
            eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
        )

        # Ensure the covariance matrix is symmetric
        cov_matrix_adjusted = (cov_matrix_adjusted + cov_matrix_adjusted.T) / 2

        # Inform the user about the adjustment
        st.info(
            "Adjusted covariance matrix to be positive definite by correcting negative eigenvalues."
        )

        return cov_matrix_adjusted

    else:
        # Inform the user
        st.info("Covariance matrix is PD.")

        return cov_matrix


def optimize_portfolio_PyPortfolioOpt(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
    risk_free_rate,
    include_risk_free_asset,
    risk_aversion,
):
    # mean_returns = expected_returns.mean_historical_return(
    #     data, frequency=12
    # )  # Annualized returns

    # Calculate expected returns and covariance matrix
    returns = data.pct_change().dropna()

    # Remove infinite values and assets with zero variance
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns.dropna(axis=1, how="any", inplace=True)
    returns = returns.loc[:, returns.std() > 0]

    mean_returns = returns.mean() * 12

    cov_matrix = returns.cov() * 12

    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        cov_matrix = risk_models.fix_nonpositive_semidefinite(
            cov_matrix
        )  # Annualized covariance

    # Adjust covariance matrix to be positive definite
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )

    st.write(f"Installed solvers : {cp.installed_solvers()}")

    # # Add small value to diagonal to ensure positive definiteness
    # epsilon = 1e-1
    # cov_matrix_np = cov_matrix.values + epsilon * np.eye(len(cov_matrix))
    # cov_matrix = pd.DataFrame(
    #     cov_matrix_np, index=returns.columns, columns=returns.columns
    # )

    if leverage_limit:
        # Set weight bounds
        if long_only:
            weight_bounds = (
                max(min_weight, 0.0),
                min(max_weight, leverage_limit_value),
            )
        else:
            weight_bounds = (
                max(min_weight, -leverage_limit_value),
                min(max_weight, leverage_limit_value),
            )
    else:
        # Set weight bounds
        if long_only:
            weight_bounds = (max(min_weight, 0.0), min(max_weight, 1.0))
        else:
            weight_bounds = (
                max(min_weight, -1),
                min(max_weight, 1),
            )

    # Prepare result similar to scipy.optimize result
    class Result:
        pass

    result = Result()

    solvers_installed = ["OSQP", "ECOS", "ECOS_BB", "SCS", "CLARABEL", "SCIPY"]

    for solver in solvers_installed:
        # Objective functions
        try:
            st.info(f"Trying solver: {solver}")

            # Initialize Efficient Frontier
            ef = EfficientFrontier(
                mean_returns,
                cov_matrix_adjusted,
                weight_bounds=weight_bounds,
                solver=solver,
            )

            # Add leverage limit constraint
            if leverage_limit:
                ef.add_constraint(lambda w: cp.sum(w) <= leverage_limit_value)
                ef.add_constraint(lambda w: cp.sum(w) >= 1)
            else:
                ef.add_constraint(lambda w: cp.sum(w) == 1)

            if include_risk_free_asset:
                # Tangency Portfolio: Maximize Sharpe Ratio
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            else:
                # Maximize Utility Function
                ef.max_quadratic_utility(risk_aversion=risk_aversion)

            # Extract weights
            cleaned_weights = ef.clean_weights()
            weights = pd.Series(cleaned_weights).reindex(assets)

            result.x = weights.values
            result.success = True
            result.status = "Optimization Succeeded"
            result.fun = ef.portfolio_performance(verbose=False)[
                1
            ]  # Portfolio volatility

            # Return result, mean returns, and covariance matrix
            return result, mean_returns, cov_matrix_adjusted

        except Exception as e:
            st.error(f"Solver {solver} failed: {e}")
            # Move on to the next solver if the current one fails

    # If all solvers fail
    result.x = None
    result.success = False
    result.status = "All solvers failed."
    result.fun = None

    # Return failure result after all solvers fail
    return result, mean_returns, cov_matrix_adjusted


def optimize_portfolio_qp(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
    risk_free_rate,
    include_risk_free_asset,
    risk_aversion,
):

    # Calculate returns, mean returns, and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12  # Annualized mean returns
    cov_matrix = returns.cov() * 12  # Annualized covariance matrix
    num_assets = len(mean_returns)
    assets = mean_returns.index.tolist()

    # # Regularize the covariance matrix to make it positive semidefinite
    # cov_matrix_np = cov_matrix.values
    # epsilon = 1e-8  # Small positive value
    # cov_matrix_np += epsilon * np.eye(num_assets)
    # cov_matrix = pd.DataFrame(
    #     cov_matrix_np, index=cov_matrix.index, columns=cov_matrix.columns
    # )

    # Define variables
    w = cp.Variable(num_assets)

    # Objective functions
    if include_risk_free_asset:
        # Tangency Portfolio: Maximize Sharpe Ratio
        # Since maximizing Sharpe Ratio directly is non-convex, we can reformulate
        # the problem to minimize portfolio variance for a unit of excess return
        # (mu - rf)^T w = 1
        # Minimize w^T Σ w

        # Define the objective function: Minimize portfolio variance
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraints = []

        # Excess return over risk-free rate equals 1
        constraints.append((mean_returns.values - risk_free_rate) @ w == 1)

    else:
        # Maximize Utility Function: Maximize expected return minus risk aversion times variance
        portfolio_return = mean_returns.values @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        objective = cp.Maximize(utility)

        # Constraints
        constraints = []

    # Sum of weights equals 1
    constraints.append(cp.sum(w) >= 1)

    # Leverage limit constraint
    constraints.append(cp.sum(w) <= leverage_limit_value)

    # Weight bounds
    if long_only:
        constraints.append(w >= max(min_weight, 0.0))
        constraints.append(w <= min(max_weight, leverage_limit_value))
    else:
        constraints.append(w >= -1)
        constraints.append(w <= 1)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None, mean_returns, cov_matrix

    # Check if the optimization was successful
    if prob.status not in ["infeasible", "unbounded"]:

        # Extract weights
        weights = w.value
        weights = pd.Series(weights, index=assets)

        # Prepare result similar to scipy.optimize result
        class Result:
            pass

        result = Result()
        result.x = weights.values
        result.success = True
        result.status = prob.status
        result.fun = prob.value

        return result, mean_returns, cov_matrix
    else:
        st.error(f"Optimization failed. Problem status: {prob.status}")
        return None, mean_returns, cov_matrix


# Optimization function
def optimize_portfolio(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
    risk_free_rate,
    include_risk_free_asset,
    risk_aversion,
):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    num_assets = len(mean_returns)
    initial_weights = num_assets * [
        1.0 / num_assets,
    ]

    # Constraints
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if leverage_limit:
        constraints = [
            {"type": "ineq", "fun": lambda x: leverage_limit_value - np.sum(x)},
            {"type": "ineq", "fun": lambda x: np.sum(x) - 1},
        ]

    # Bounds
    if long_only:
        bounds = tuple(
            (max(min_weight, 0.0), min(max_weight, 1.0)) for _ in range(num_assets)
        )
    else:
        bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

    # Objective functions
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    def negative_utility(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        utility = portfolio_return - 0.5 * risk_aversion * (portfolio_volatility**2)
        return -utility

    # Choose the appropriate objective function
    if include_risk_free_asset:
        objective_function = neg_sharpe_ratio
    else:
        objective_function = negative_utility

    # Progress bar
    progress_bar = st.progress(0)
    iteration_container = st.empty()

    max_iterations = (
        1000  # Set maximum number of iterations for estimation if taking too long
    )

    iteration_counter = {"n_iter": 0}

    # Callback function to update progress
    def callbackF(xk):
        iteration_counter["n_iter"] += 1
        progress = iteration_counter["n_iter"] / max_iterations
        progress_bar.progress(min(progress, 1.0))
        iteration_container.text(f"Iteration: {iteration_counter['n_iter']}")

    # Estimated time indicator
    st.info("Estimated time to complete optimization: depends on data and constraints.")

    with st.spinner("Optimization in progress..."):
        start_time = time.time()
        result = minimize(
            objective_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations},
            callback=callbackF,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

    progress_bar.empty()
    iteration_container.empty()

    st.success(f"Optimization completed in {elapsed_time:.2f} seconds")
    return result, mean_returns, cov_matrix


# -------------------------------
# 6. Efficient Frontier Calculation
# -------------------------------


# 6. Efficient Frontier Calculation Using PyPortfolioOpt
def calculate_efficient_frontier_pypfopt(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    # Set weight bounds
    if long_only:
        weight_bounds = (
            max(min_weight_value, 0.0),
            min(max_weight_value, leverage_limit_value),
        )
    else:
        weight_bounds = (
            -1,
            1,
        )

    # Initialize Efficient Frontier
    ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)

    # Add leverage limit constraint
    ef.add_constraint(lambda w: cp.sum(w) <= leverage_limit_value)
    ef.add_constraint(lambda w: cp.sum(w) >= 1)

    # Generate Efficient Frontier
    target_returns = np.linspace(
        -mean_returns.max(),
        mean_returns.max(),
        50,
    )
    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []

    for ret in stqdm(target_returns, desc="PyPortfolioOpt frontier computation..."):
        ef_copy = ef.deepcopy()
        ef_copy.efficient_return(target_return=ret)
        weights = ef_copy.weights
        frontier_weights.append(weights)
        performance = ef_copy.portfolio_performance(risk_free_rate=risk_free_rate)
        frontier_returns.append(performance[0])
        frontier_volatility.append(performance[1])

    return frontier_volatility, frontier_returns, frontier_weights


def calculate_efficient_frontier_qp(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    num_assets = len(mean_returns)
    cov_matrix = cov_matrix.values
    mean_returns = mean_returns.values

    # # Regularize the covariance matrix to make it positive semidefinite
    # epsilon = 1e-7  # Small positive value
    # cov_matrix_np += epsilon * np.eye(num_assets)
    # cov_matrix = pd.DataFrame(
    #     cov_matrix_np, index=cov_matrix.index, columns=cov_matrix.columns
    # )

    # Define variables
    w = cp.Variable((num_assets, 1))
    portfolio_return = mean_returns.T @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)

    if leverage_limit:
        # Leverage limit constraint
        # Constraints
        constraints = [cp.sum(w) >= 1]
        constraints += [cp.sum(w) <= leverage_limit_value]
    else:
        constraints = [cp.sum(w) == 1]

    # Weight bounds
    if long_only:
        constraints += [
            w >= max(min_weight_value, 0.0),
            w <= min(max_weight_value, leverage_limit_value),
        ]
    else:
        constraints += [
            w >= -1,
            w <= 1,
        ]

    # Target returns for the efficient frontier
    target_returns = np.linspace(
        -mean_returns.max(),
        mean_returns.max() * 3,
        50,
    )

    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []

    for target_return in stqdm(target_returns, desc="QP Frontier computation..."):
        # Objective: Minimize variance
        objective = cp.Minimize(portfolio_variance)

        # Constraints for target return
        constraints_with_return = constraints + [portfolio_return == target_return]

        # Problem
        prob = cp.Problem(objective, constraints_with_return)

        # Solve the problem
        prob.solve(solver=cp.SCS)

        if prob.status not in ["infeasible", "unbounded"]:
            vol = np.sqrt(portfolio_variance.value)[0]  # Annualized volatility
            frontier_volatility.append(vol)
            frontier_returns.append(target_return)
            frontier_weights.append(w.value.flatten())
        else:
            st.warning(
                f"Optimization failed for target return {target_return:.2%}. Status: {prob.status}"
            )
            continue

    return frontier_volatility, frontier_returns, frontier_weights


# Efficient Frontier Calculation
def calculate_efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    if leverage_limit:
        target_returns = np.linspace(
            mean_returns.min(),
            mean_returns.max() * leverage_limit_value * 12 * 2.5,
            50,
        )
    else:
        target_returns = np.linspace(
            -mean_returns.max() * 12 * leverage_limit_value,
            mean_returns.max() * 12 * leverage_limit_value,
            50,
        )
    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []

    num_assets = len(mean_returns)

    for idx, target_return in enumerate(
        stqdm(target_returns, desc="Computing the frontier... ")
    ):
        # Constraints for the optimization
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Sum of weights equals 1
            {
                "type": "eq",
                "fun": lambda x: np.sum(x * mean_returns * 12)
                - target_return,  # Target return constraint
            },
        ]

        # Leverage limit constraint
        if leverage_limit:
            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x: leverage_limit_value
                    - np.sum(x),  # Sum of weights <= leverage limit
                },
                {"type": "ineq", "fun": lambda x: np.sum(x) - 1},  # Sum of weights >= 1
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x * mean_returns * 12)
                    - target_return,  # Target return constraint --> portfolio return = target return
                },
            ]

        # Bounds
        if long_only:
            # Apply minimum and maximum weight constraints
            bounds = tuple(
                (max(min_weight_value, 0.0), min(max_weight_value, 1.0))
                for _ in range(num_assets)
            )
        else:
            # Allow short selling within a limit of 1 per asset
            bounds = tuple((-1, 1) for _ in range(num_assets))

        # Progress bar
        progress_bar = st.progress(0)
        iteration_container = st.empty()

        max_iterations = (
            10  # Set maximum number of iterations for estimation if taking too long
        )

        iteration_counter = {"n_iter": 0}

        # Callback function to update progress
        def callbackF(xk):
            iteration_counter["n_iter"] += 1
            progress = iteration_counter["n_iter"] / max_iterations
            progress_bar.progress(min(progress, 1.0))
            iteration_container.text(f"Iteration: {iteration_counter['n_iter']}")

        # Optimization
        with st.spinner("Optimization in progress..."):
            start_time = time.time()
            result = minimize(
                lambda x: np.sqrt(
                    np.dot(x.T, np.dot(cov_matrix * 12, x))
                ),  # Minimize volatility
                num_assets * [1.0 / num_assets],  # Initial guess
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations},
                callback=callbackF,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time

        progress_bar.empty()
        iteration_container.empty()

        if result.success:
            st.success(f"Optimization completed in {elapsed_time:.2f} seconds")
            frontier_volatility.append(result.fun)
            frontier_returns.append(target_return)
            frontier_weights.append(result.x)
        elif result.status == 9:
            st.warning(
                f"Optimization for target return {target_return:.2%} reached maximum iterations."
            )
            frontier_volatility.append(result.fun)
            frontier_returns.append(target_return)
            frontier_weights.append(result.x)
        else:
            # Handle optimization failure
            st.warning(f"Optimization failed for target return {target_return:.2%}")
            pass

    return frontier_volatility, frontier_returns, frontier_weights


# -------------------------------
# 7. Sampling Methods Implementation
# -------------------------------

# --- Adjusted Dirichlet Sampling ---


def generate_biased_random_portfolios(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    num_portfolios,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
    bias_method="expected_returns",
    lambda_scale=100,
):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = np.zeros((num_portfolios, num_assets))
    accepted_portfolios = 0  # Index for accepted portfolios

    # Compute alpha parameters for Dirichlet distribution
    if bias_method == "expected_returns":
        alpha = mean_returns.values.copy()
        alpha = alpha - alpha.min() + 1e-6  # Shift to make all values positive
        alpha = alpha / alpha.sum()
    elif bias_method == "inverse_variance":
        variances = np.diag(cov_matrix)
        alpha = 1 / variances
        alpha = alpha - alpha.min() + 1e-6
        alpha = alpha / alpha.sum()
    elif bias_method == "combined":
        variances = np.diag(cov_matrix)
        alpha = mean_returns.values / variances
        alpha = alpha - alpha.min() + 1e-6
        alpha = alpha / alpha.sum()
    else:
        raise ValueError(
            "Invalid bias_method. Choose 'expected_returns', 'inverse_variance', or 'combined'."
        )

    # Scale alpha parameters
    alpha = alpha * lambda_scale

    count = 0

    for i in stqdm(range(num_portfolios), desc="Dirichlet sampling..."):

        # Generate weights using Dirichlet distribution with adjusted alpha
        weights = np.random.dirichlet(alpha)
        s = np.random.uniform(1, leverage_limit_value)
        weights *= s  # Scale weights

        # Apply weight bounds
        weights = np.clip(weights, -1, 1)

        # # Normalize weights to sum to 1
        # weights /= np.sum(weights)

        # Check leverage limit
        if np.sum(weights) > leverage_limit_value or np.sum(weights) < 1:
            count += 1
            continue  # Skip this portfolio

        # Apply long-only constraint if necessary
        if long_only and np.any(weights < 0):
            count += 1
            continue  # Skip portfolios with negative weights

        # Calculate portfolio performance
        portfolio_return = np.sum(mean_returns * weights) * 12  # Annualized return
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix * 12, weights))
        )  # Annualized volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        weights_record[i, :] = weights

        accepted_portfolios += 1

    st.write(f"Skipped portfolios from Dirichet: {count}")

    # Truncate arrays to include only accepted portfolios
    results = results[:, :accepted_portfolios]
    weights_record = weights_record[:accepted_portfolios, :]

    return results, weights_record


# --- Latin Hypercube Sampling (LHS) ---


def generate_lhs_random_portfolios(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    num_portfolios,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = np.zeros((num_portfolios, num_assets))
    accepted_portfolios = 0  # Index for accepted portfolios

    # Create a Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=num_assets)
    sample = sampler.random(n=num_portfolios)

    # Scale samples to the desired weight bounds
    if long_only:
        lower_bounds = np.full(num_assets, max(min_weight_value, 0.0))
        upper_bounds = np.full(num_assets, min(max_weight_value, leverage_limit_value))
    else:
        lower_bounds = np.full(num_assets, -1)
        upper_bounds = np.full(num_assets, 1)

    # Transform samples to the desired range
    weights_array = qmc.scale(sample, lower_bounds, upper_bounds)

    count = 0
    for i in stqdm(range(num_portfolios), desc="LHS sampling..."):
        weights = weights_array[i, :]

        # # Normalize weights to sum to 1
        # weights /= np.sum(weights)

        # Check leverage limit
        if np.sum(weights) > leverage_limit_value or np.sum(weights) < 1:
            count += 1
            continue  # Skip this portfolio

        # Apply long-only constraint if necessary
        if long_only and np.any(weights < 0):
            continue  # Skip portfolios with negative weights

        # Calculate portfolio performance
        portfolio_return = np.sum(mean_returns * weights)  # Annualized return
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )  # Annualized volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        weights_record[accepted_portfolios, :] = weights

        accepted_portfolios += 1  # Increment index

    st.write(f"Skipped portfolios from LHS: {count}")

    # Truncate arrays to include only accepted portfolios
    results = results[:, :accepted_portfolios]
    weights_record = weights_record[:accepted_portfolios, :]

    return results, weights_record


# -------------------------------
# 8. Plotting Function Incorporating Both Sampling Methods
# -------------------------------


# Efficient Frontier Plotting Function
def plot_efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    weights_optimal,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
    tangency_weights,
    num_portfolios=5000,
):
    # Calculate the efficient frontier with updated constraints
    frontier_volatility, frontier_returns, frontier_weights = (
        calculate_efficient_frontier_qp(
            mean_returns,
            cov_matrix,
            risk_free_rate,
            include_risk_free_asset,
            long_only,
            leverage_limit_value,
            min_weight_value,
            max_weight_value,
        )
    )

    # Check tangency weights
    if np.sum(tangency_weights) > leverage_limit_value or np.sum(tangency_weights) < 1:
        st.write("Tangency portfolio doesn't meet the constraints ")
    else:
        st.write("Tangency portfolio fit the constraints")

    if np.min(tangency_weights) < -1 or np.max(tangency_weights) > 1:
        st.write("Tangency portfolio doesn't fit the individual asset leverage limit")
    else:
        st.write("Tangency portfolio fit the individual asset leverage limit")

    # # Generate portfolios using Adjusted Dirichlet Sampling
    # st.info("Generating portfolios using Adjusted Dirichlet Sampling...")
    # results_dirichlet, _ = generate_biased_random_portfolios(
    #     mean_returns,
    #     cov_matrix,
    #     risk_free_rate,
    #     num_portfolios,
    #     long_only,
    #     leverage_limit_value,
    #     min_weight_value,
    #     max_weight_value,
    #     bias_method="expected_returns",  # You can change to 'inverse_variance' or 'combined'
    #     lambda_scale=10,  # Adjust as needed
    # )

    # Generate portfolios using Latin Hypercube Sampling
    st.info("Generating portfolios using Latin Hypercube Sampling...")
    results_lhs, _ = generate_lhs_random_portfolios(
        mean_returns,
        cov_matrix,
        risk_free_rate,
        num_portfolios,
        long_only,
        leverage_limit_value,
        min_weight_value,
        max_weight_value,
    )

    # Plotting
    plt.figure(figsize=(10, 7))

    # # Plot Adjusted Dirichlet Sampling portfolios
    # plt.scatter(
    #     results_dirichlet[0],
    #     results_dirichlet[1],
    #     c=results_dirichlet[2],
    #     cmap="viridis",
    #     s=2,
    #     alpha=0.4,
    #     label="Adjusted Dirichlet Portfolios",
    # )

    # Plot Latin Hypercube Sampling portfolios
    plt.scatter(
        results_lhs[0],
        results_lhs[1],
        c=results_lhs[2],
        cmap="plasma",
        s=2,
        alpha=0.4,
        label="LHS Portfolios",
    )

    # plt.colorbar(label="Sharpe Ratio")
    plt.plot(
        frontier_volatility,
        frontier_returns,
        "r--",
        linewidth=3,
        label="Efficient Frontier",
    )

    if include_risk_free_asset:

        # Plot the Capital Market Line and Tangency Portfolio
        tangency_weights = weights_optimal
        tangency_return = np.sum(mean_returns * tangency_weights)
        tangency_volatility = np.sqrt(
            np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights))
        )

        # Plot the Capital Market Line
        cml_x = [0, tangency_volatility]
        cml_y = [risk_free_rate, tangency_return]
        plt.plot(
            cml_x, cml_y, color="green", linestyle="--", label="Capital Market Line"
        )

        # Highlight the tangency portfolio
        plt.scatter(
            tangency_volatility,
            tangency_return,
            marker="*",
            color="red",
            s=500,
            label="Tangency Portfolio",
        )
        # else:
        #     st.warning("Failed to compute the tangency portfolio.")
    else:
        # Highlight the optimal portfolio
        portfolio_return = np.sum(mean_returns * weights_optimal)
        portfolio_volatility = np.sqrt(
            np.dot(weights_optimal.T, np.dot(cov_matrix, weights_optimal))
        )
        plt.scatter(
            portfolio_volatility,
            portfolio_return,
            marker="*",
            color="red",
            s=500,
            label="Optimal Portfolio",
        )

    plt.title("Efficient Frontier with Random Portfolios")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Expected Returns")
    plt.legend()
    st.pyplot(plt)


# -------------------------------
# 9. Main Application Logic
# -------------------------------


if st.button("Run Optimization"):
    tangency_result, tangency_mean_returns, tangency_cov_matrix = (
        optimize_portfolio_PyPortfolioOpt(
            data,
            long_only,
            min_weight_value,
            max_weight_value,
            leverage_limit_value,
            risk_free_rate,
            include_risk_free_asset,
            risk_aversion,
        )
    )
    weights = pd.Series(tangency_result.x, index=assets)
    st.session_state["optimization_run"] = True
    st.session_state["weights"] = weights
    st.session_state["mean_returns"] = tangency_mean_returns
    st.session_state["cov_matrix"] = tangency_cov_matrix

    # Display optimization results
    # st.write(weights.apply(lambda x: f"{x:.2%}"))

    # Calculate portfolio performance
    portfolio_return = np.sum(tangency_mean_returns * weights)  # Annualized return
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(tangency_cov_matrix, weights))
    )  # Annualized volatility

    if include_risk_free_asset:
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Calculate allocation between risk-free asset and tangency portfolio
        allocation_tangency = (portfolio_return - risk_free_rate) / (
            risk_aversion * (portfolio_volatility**2)
        )
        allocation_tangency = min(max(allocation_tangency, 0), sum(weights))
        allocation_risk_free = max(sum(weights) - allocation_tangency, 0)

        st.subheader("Portfolio Performance with Risk-Free Asset:")
        st.write(f"Expected Annual Return: {portfolio_return:.2%}")
        st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
        st.write(f"Max tangency mean returns: {tangency_mean_returns.max()}")
        st.write(f"Max tangency weights: {weights.max()}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        st.write(f"Invest {allocation_tangency * 100:.2f}% in the tangency portfolio.")
        st.write(f"Invest {allocation_risk_free * 100:.2f}% in the risk-free asset.")

        # Show the allocation
        allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
        # Create a mapping of ISIN to Company name from static_data
        isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))

        # Replace ISIN in allocation_df with the corresponding company names
        allocation_df["ISIN"] = allocation_df["ISIN"].map(isin_to_company)

        # Optionally rename the column to reflect the new data
        allocation_df.rename(columns={"ISIN": "Company"})

        st.subheader("Tangency Portfolio Weights:")
        st.write(allocation_df)

    else:
        # Calculate Sharpe Ratio
        sharpe_ratio = portfolio_return / portfolio_volatility

        st.subheader("Portfolio Performance without Risk-Free Asset:")
        st.write(f"Expected Annual Return: {portfolio_return:.2%}")
        st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Show the allocation
        allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
        # Create a mapping of ISIN to Company name from static_data
        isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))

        # Replace ISIN in allocation_df with the corresponding company names
        allocation_df["ISIN"] = allocation_df["ISIN"].map(isin_to_company)

        # Optionally rename the column to reflect the new data
        allocation_df.rename(columns={"ISIN": "Company"})

        st.write("Optimal Portfolio Allocation:")
        st.write(allocation_df)
    st.write(f"Sum of the weights: {np.sum(weights)}")
else:
    st.write('Click "Run Optimization" to compute the optimized portfolio.')

# Run the efficient frontier
if st.session_state["optimization_run"]:
    if st.button("Show Efficient Frontier"):
        # Retrieve necessary variables from session state
        weights = st.session_state["weights"]
        mean_returns = st.session_state["mean_returns"]
        cov_matrix = st.session_state["cov_matrix"]

        st.write(f"Max mean returns for plot: {mean_returns.max()}")
        num_assets = len(mean_returns)
        weights_optimal = weights.values

        plot_efficient_frontier(
            mean_returns,
            cov_matrix,
            risk_free_rate,
            include_risk_free_asset,
            weights_optimal,
            long_only,
            leverage_limit_value,
            min_weight_value,
            max_weight_value,
            weights_optimal,
            num_portfolios=50000,
        )
    else:
        st.write('Click "Show Efficient Frontier" to display the graph.')
else:
    st.write("Run the optimization first to display the efficient frontier.")

# # Efficient Frontier Plotting Function
# def plot_efficient_frontier(
#     mean_returns,
#     cov_matrix,
#     risk_free_rate,
#     include_risk_free_asset,
#     weights_optimal,
#     long_only,
#     bounds,
# ):
#     frontier_volatility, frontier_returns, frontier_weights = (
#         calculate_efficient_frontier(
#             mean_returns,
#             cov_matrix,
#             risk_free_rate,
#             include_risk_free_asset,
#             long_only,
#             bounds,
#         )
#     )

#     plt.figure(figsize=(10, 7))
#     plt.plot(frontier_volatility, frontier_returns, "b--", label="Efficient Frontier")

#     if include_risk_free_asset:
#         # Calculate the tangency portfolio
#         def neg_sharpe_ratio(weights):
#             portfolio_return = np.sum(mean_returns * weights) * 12
#             portfolio_volatility = np.sqrt(
#                 np.dot(weights.T, np.dot(cov_matrix * 12, weights))
#             )
#             sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
#             return -sharpe_ratio

#         constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
#         result = minimize(
#             neg_sharpe_ratio,
#             num_assets * [1.0 / num_assets],
#             method="SLSQP",
#             bounds=bounds,
#             constraints=constraints,
#         )

#         if result.success:
#             tangency_weights = result.x
#             tangency_return = np.sum(mean_returns * tangency_weights) * 12
#             tangency_volatility = np.sqrt(
#                 np.dot(tangency_weights.T, np.dot(cov_matrix * 12, tangency_weights))
#             )

#             # Plot the Capital Market Line
#             cml_x = [0, tangency_volatility]
#             cml_y = [risk_free_rate, tangency_return]
#             plt.plot(cml_x, cml_y, color="red", label="Capital Market Line")

#             # Highlight the tangency portfolio
#             plt.scatter(
#                 tangency_volatility,
#                 tangency_return,
#                 marker="*",
#                 color="red",
#                 s=500,
#                 label="Tangency Portfolio",
#             )
#         else:
#             st.warning("Failed to compute the tangency portfolio.")
#     else:
#         # Highlight the optimal portfolio
#         portfolio_return = np.sum(mean_returns * weights_optimal) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights_optimal.T, np.dot(cov_matrix * 12, weights_optimal))
#         )
#         plt.scatter(
#             portfolio_volatility,
#             portfolio_return,
#             marker="*",
#             color="red",
#             s=500,
#             label="Optimal Portfolio",
#         )

#     plt.title("Efficient Frontier")
#     plt.xlabel("Annualized Volatility")
#     plt.ylabel("Annualized Expected Returns")
#     plt.legend()
#     st.pyplot(plt)


# if st.button("Show Efficient Frontier"):
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
#     num_assets = len(mean_returns)
#     if "weights" in locals():
#         weights_optimal = weights.values
#     else:
#         weights_optimal = None

#     # Prepare bounds for efficient frontier calculation
#     if long_only:
#         bounds = tuple(
#             (max(min_weight_value, 0.0), min(max_weight_value, 1.0))
#             for _ in range(num_assets)
#         )
#     else:
#         bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

#     plot_efficient_frontier(
#         mean_returns,
#         cov_matrix,
#         risk_free_rate,
#         include_risk_free_asset,
#         weights_optimal,
#         long_only,
#         bounds,
#     )
# else:
#     st.write('Click "Show Efficient Frontier" to display the graph.')


# # Efficient Frontier using statistical generation
# def plot_efficient_frontier(
#     mean_returns, cov_matrix, risk_free_rate, include_risk_free_asset, weights_optimal
# ):
#     num_portfolios = 5000
#     results = np.zeros((3, num_portfolios))
#     for i in range(num_portfolios):
#         weights = np.random.dirichlet(np.ones(len(mean_returns)))
#         portfolio_return = np.sum(mean_returns * weights) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights.T, np.dot(cov_matrix * 12, weights))
#         )
#         sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
#         results[0, i] = portfolio_volatility
#         results[1, i] = portfolio_return
#         results[2, i] = sharpe_ratio

#     max_sharpe_idx = np.argmax(results[2])
#     sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

#     plt.figure(figsize=(10, 7))
#     plt.scatter(
#         results[0],
#         results[1],
#         c=results[2],
#         cmap="viridis",
#         marker="o",
#         s=10,
#         alpha=0.3,
#     )
#     plt.colorbar(label="Sharpe Ratio")
#     if include_risk_free_asset:
#         # Plot the Capital Market Line
#         cml_x = [0, sdp]
#         cml_y = [risk_free_rate, rp]
#         plt.plot(cml_x, cml_y, color="red", label="Capital Market Line")

#         # Highlight the tangency portfolio
#         plt.scatter(sdp, rp, marker="*", color="red", s=500, label="Tangency Portfolio")
#     else:
#         # Highlight the optimal portfolio
#         portfolio_return = np.sum(mean_returns * weights_optimal) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights_optimal.T, np.dot(cov_matrix * 12, weights_optimal))
#         )
#         plt.scatter(
#             portfolio_volatility,
#             portfolio_return,
#             marker="*",
#             color="red",
#             s=500,
#             label="Optimal Portfolio",
#         )

#     plt.title("Efficient Frontier")
#     plt.xlabel("Volatility (Std. Deviation)")
#     plt.ylabel("Expected Returns")
#     plt.legend()
#     st.pyplot(plt)


# if st.button("Show Efficient Frontier"):
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
#     if "weights" in locals():
#         weights_optimal = weights.values
#     else:
#         weights_optimal = None
#     plot_efficient_frontier(
#         mean_returns,
#         cov_matrix,
#         risk_free_rate,
#         include_risk_free_asset,
#         weights_optimal,
#     )
# else:
#     st.write('Click "Show Efficient Frontier" to display the graph.')

# if __name__ == "__main__":
#     weights = optimize_portfolio(data, True, 0, 1, 1, 0.01)
#     print(weights)
