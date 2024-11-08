import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import cvxpy as cp
import scipy.optimize as sco
from pypfopt import (
    EfficientFrontier,
    expected_returns,
    risk_models,
    CLA,
    objective_functions,
)
import time

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


# Initialize current page in session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Quiz"

data = pd.read_csv("Cleaned_df.csv", index_col="Date")
static_data = pd.read_excel(
    r"C:\Users\aleks\Downloads\Static.xlsx"
)
assets = data.columns.tolist()


def main():
    if st.session_state["current_page"] == "Quiz":
        risk_aversion_quiz()
    elif st.session_state["current_page"] == "Data Visualization":
        data_visualization_page()
    elif st.session_state["current_page"] == "Optimization":
        optimization_page()
    elif st.session_state["current_page"] == "Efficient Frontier":
        efficient_frontier_page()
    else:
        st.write("Page not found.")


# -------------------------------
# 2. Risk Aversion Quiz
# -------------------------------


# Risk aversion quiz using a form
def risk_aversion_quiz():
    st.title("Portfolio Optimization Application")
    st.header("Risk Aversion Quiz")
    with st.form(key="quiz_form"):
        # Initialize score
        score_original = 0

        # Question 1
        q1 = st.radio(
            "1. In general, how would your best friend describe you as a risk taker?",
            (
                "a. A real gambler",
                "b. Willing to take risks after completing adequate research",
                "c. Cautious",
                "d. A real risk avoider",
            ),
            key="q1",
        )

        # Question 2
        q2 = st.radio(
            "2. You are on a TV game show and can choose one of the following, which would you take?",
            (
                "a. $1,000 in cash",
                "b. A 50% chance at winning $5,000",
                "c. A 25% chance at winning $10,000",
                "d. A 5% chance at winning $100,000",
            ),
            key="q2",
        )

        # Question 3
        q3 = st.radio(
            "3. You have just finished saving for a 'once-in-a-lifetime' vacation. Three weeks before you plan to leave, you lose your job. You would:",
            (
                "a. Cancel the vacation",
                "b. Take a much more modest vacation",
                "c. Go as scheduled, reasoning that you need the time to prepare for a job search",
                "d. Extend your vacation, because this might be your last chance to go first-class",
            ),
            key="q3",
        )

        # Question 4
        q4 = st.radio(
            "4. If you unexpectedly received $20,000 to invest, what would you do?",
            (
                "a. Deposit it in a bank account, money market account, or an insured CD",
                "b. Invest it in safe high quality bonds or bond mutual funds",
                "c. Invest it in stocks or stock mutual funds",
            ),
            key="q4",
        )

        # Question 5
        q5 = st.radio(
            "5. In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
            (
                "a. Not at all comfortable",
                "b. Somewhat comfortable",
                "c. Very comfortable",
            ),
            key="q5",
        )

        # Question 6
        q6 = st.radio(
            "6. When you think of the word 'risk,' which of the following words comes to mind first?",
            (
                "a. Loss",
                "b. Uncertainty",
                "c. Opportunity",
                "d. Thrill",
            ),
            key="q6",
        )

        # Question 7
        q7 = st.radio(
            "7. Some experts are predicting prices of assets such as gold, jewels, collectibles, and real estate (hard assets) to increase in value; bond prices may fall. However, experts tend to agree that government bonds are relatively safe. Most of your investment assets are now in high-interest government bonds. What would you do?",
            (
                "a. Hold the bonds",
                "b. Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets",
                "c. Sell the bonds and put the total proceeds into hard assets",
                "d. Sell the bonds, put all the money into hard assets, and borrow additional money to buy more",
            ),
            key="q7",
        )

        # Question 8
        q8 = st.radio(
            "8. Given the best and worst-case returns of the four investment choices below, which would you prefer?",
            (
                "a. $200 gain best case; $0 gain/loss worst case",
                "b. $800 gain best case; $200 loss worst case",
                "c. $2,600 gain best case; $800 loss worst case",
                "d. $4,800 gain best case; $2,400 loss worst case",
            ),
            key="q8",
        )

        # Question 9
        q9 = st.radio(
            "9. In addition to whatever you own, you have been given $1,000. You are now asked to choose between:",
            (
                "a. A sure gain of $500",
                "b. A 50% chance to gain $1,000 and a 50% chance to gain nothing",
            ),
            key="q9",
        )

        # Question 10
        q10 = st.radio(
            "10. In addition to whatever you own, you have been given $2,000. You are now asked to choose between:",
            (
                "a. A sure loss of $500",
                "b. A 50% chance to lose $1,000 and a 50% chance to lose nothing",
            ),
            key="q10",
        )

        # Question 11
        q11 = st.radio(
            "11. Suppose a relative left you an inheritance of $100,000, stipulating in the will that you invest ALL the money in ONE of the following choices. Which one would you select?",
            (
                "a. A savings account or money market mutual fund",
                "b. A mutual fund that owns stocks and bonds",
                "c. A portfolio of 15 common stocks",
                "d. Commodities like gold, silver, and oil",
            ),
            key="q11",
        )

        # Question 12
        q12 = st.radio(
            "12. If you had to invest $20,000, which of the following investment choices would you find most appealing?",
            (
                "a. 60% in low-risk investments, 30% in medium-risk investments, 10% in high-risk investments",
                "b. 30% in low-risk investments, 40% in medium-risk investments, 30% in high-risk investments",
                "c. 10% in low-risk investments, 40% in medium-risk investments, 50% in high-risk investments",
            ),
            key="q12",
        )

        # Question 13
        q13 = st.radio(
            "13. Your trusted friend and neighbor, an experienced geologist, is putting together a group of investors to fund an exploratory gold mining venture. The venture could pay back 50 to 100 times the investment if successful. If the mine is a bust, the entire investment is worthless. Your friend estimates the chance of success is only 20%. If you had the money, how much would you invest?",
            (
                "a. Nothing",
                "b. One month's salary",
                "c. Three months' salary",
                "d. Six months' salary",
            ),
            key="q13",
        )

        ### Demographic Questions (Q14 - Q20)
        st.header("Demographic Information (Scored)")

        # Q14. Gender
        q14_gender = st.selectbox("14. What is your gender?", ["Male", "Female", "Other"], key="q14")

        # Q15. Age
        q15_age = st.number_input("15. What is your current age in years?", min_value=0, max_value=120, key="q15")

        # Q16. Marital Status
        q16_marital_status = st.selectbox("16. What is your marital status?", ["Single", "Living with significant other", "Married", "Separated/Divorced", "Widowed", "Shared living arrangement"], key="q16")

        # Q17. Education
        q17_education = st.selectbox("17. What is the highest level of education you have completed?", ["Associate's degree or less", "Some college", "High school diploma", "Some high school or less", "Bachelor's degree", "Graduate or professional degree"], key="q17")

        # Q18. Household Income
        q18_income = st.selectbox("18. What is your household's approximate annual gross income before taxes?", ["Less than $25,000", "$25,000 - $49,999", "$50,000 - $74,999", "$75,000 - $99,999", "$100,000 or more"], key="q18")

        # Q19. Investment Allocation
        st.write("Approximately what percentage of your personal and retirement savings and investments are in the following categories? (Total must be 100%)")
        q19_cash = st.number_input("Cash (e.g., savings accounts, CDs)", min_value=0, max_value=100, key="q19_cash")
        q19_bonds = st.number_input("Fixed income (e.g., bonds)", min_value=0, max_value=100, key="q19_bonds")
        q19_equities = st.number_input("Equities (e.g., stocks)", min_value=0, max_value=100, key="q19_equities")
        q19_other = st.number_input("Other (e.g., gold, collectibles)", min_value=0, max_value=100, key="q19_other")

        # Ensure the total percentages sum to 100%
        total_allocation = q19_cash + q19_bonds + q19_equities + q19_other
        if total_allocation != 100:
            st.error("The total allocation percentages must sum to 100%.")

        # Q20. Investment Decision-Making
        q20_decision_maker = st.selectbox("20. Who is responsible for investment allocation decisions in your household?", ["I make my own investment decisions", "I rely on the advice of a professional", "I do not have investment assets"], key="q20")

        submit_quiz = st.form_submit_button("Submit Quiz")

    if submit_quiz:

        ### Grable & Lytton 13 Questions (Q1 - Q13)

        # Question 1 scoring
        score_original += {
            "a. A real gambler": 4,
            "b. Willing to take risks after completing adequate research": 3,
            "c. Cautious": 2,
            "d. A real risk avoider": 1,
        }[q1]

        # Question 2 scoring
        score_original += {
            "a. $1,000 in cash": 1,
            "b. A 50% chance at winning $5,000": 2,
            "c. A 25% chance at winning $10,000": 3,
            "d. A 5% chance at winning $100,000": 4,
        }[q2]

        # Question 3 scoring
        score_original += {
            "a. Cancel the vacation": 1,
            "b. Take a much more modest vacation": 2,
            "c. Go as scheduled, reasoning that you need the time to prepare for a job search": 3,
            "d. Extend your vacation, because this might be your last chance to go first-class": 4,
        }[q3]

        # Question 4 scoring
        score_original += {
            "a. Deposit it in a bank account, money market account, or an insured CD": 1,
            "b. Invest it in safe high quality bonds or bond mutual funds": 2,
            "c. Invest it in stocks or stock mutual funds": 3,
        }[q4]

        # Question 5 scoring
        score_original += {
            "a. Not at all comfortable": 1,
            "b. Somewhat comfortable": 2,
            "c. Very comfortable": 3,
        }[q5]

        # Question 6 scoring
        score_original += {
            "a. Loss": 1,
            "b. Uncertainty": 2,
            "c. Opportunity": 3,
            "d. Thrill": 4,
        }[q6]

        # Question 7 scoring
        score_original += {
            "a. Hold the bonds": 1,
            "b. Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets": 2,
            "c. Sell the bonds and put the total proceeds into hard assets": 3,
            "d. Sell the bonds, put all the money into hard assets, and borrow additional money to buy more": 4,
        }[q7]

        # Question 8 scoring
        score_original += {
            "a. $200 gain best case; $0 gain/loss worst case": 1,
            "b. $800 gain best case; $200 loss worst case": 2,
            "c. $2,600 gain best case; $800 loss worst case": 3,
            "d. $4,800 gain best case; $2,400 loss worst case": 4,
        }[q8]

        # Question 9 scoring
        q9_score = {
            "a. A sure gain of $500": 1,
            "b. A 50% chance to gain $1,000 and a 50% chance to gain nothing": 3,
        }[q9]

        # Question 10 scoring
        q10_score = {
            "a. A sure loss of $500": 1,
            "b. A 50% chance to lose $1,000 and a 50% chance to lose nothing": 3,
        }[q10]

        # Average score for questions 9 and 10
        average_q9_q10 = (q9_score + q10_score) / 2
        score_original += average_q9_q10

        # Question 11 scoring
        score_original += {
            "a. A savings account or money market mutual fund": 1,
            "b. A mutual fund that owns stocks and bonds": 2,
            "c. A portfolio of 15 common stocks": 3,
            "d. Commodities like gold, silver, and oil": 4,
        }[q11]

        # Question 12 scoring
        score_original += {
            "a. 60% in low-risk investments, 30% in medium-risk investments, 10% in high-risk investments": 1,
            "b. 30% in low-risk investments, 40% in medium-risk investments, 30% in high-risk investments": 2,
            "c. 10% in low-risk investments, 40% in medium-risk investments, 50% in high-risk investments": 3,
        }[q12]

        # Question 13 scoring
        score_original += {
            "a. Nothing": 1,
            "b. One month's salary": 2,
            "c. Three months' salary": 3,
            "d. Six months' salary": 4,
        }[q13]

        # Scoring Demographic Questions
        score_demographic = 0

        # Gender (5 points)
        if q14_gender == "Male":
            score_demographic += 5.0
        else:
            score_demographic += 0

        # Age (3 points)
        if q15_age < 25:
            score_demographic += 1.37
        elif 25 <= q15_age <= 34:
            score_demographic += 2.44
        elif 35 <= q15_age <= 44:
            score_demographic += 3.00
        elif 45 <= q15_age <= 54:
            score_demographic += 2.10
        elif 55 <= q15_age <= 64:
            score_demographic += 0.78
        elif 65 <= q15_age <= 74:
            score_demographic += 0
        elif q15_age >= 75:
            score_demographic += 1.75

        # Marital Status (1.60 points)
        marital_points = {
            "Widowed": 0,
            "Single": 1.00,
            "Living with significant other": 1.09,
            "Married": 1.36,
            "Shared living arrangement": 1.60,
            "Separated/Divorced": 0.75  # Assuming an average point
        }
        score_demographic += marital_points.get(q16_marital_status, 0)

        # Education (2.96 points)
        education_points = {
            "Associate's degree or less": 0,
            "Some college": 0.15,
            "High school diploma": 0.64,
            "Some high school or less": 0.87,
            "Bachelor's degree": 2.36,
            "Graduate or professional degree": 2.96
        }
        score_demographic += education_points.get(q17_education, 0)

        # Household Income (3.73 points)
        income_points = {
            "Less than $25,000": 0.89,
            "$25,000 - $49,999": 0,
            "$50,000 - $74,999": 1.25,
            "$75,000 - $99,999": 2.03,
            "$100,000 or more": 3.73
        }
        score_demographic += income_points.get(q18_income, 0)

        # Scoring Question 19
        MaxPoints_Q19 = 5  # Maximum points for Q19
        numerator = (q19_equities + q19_other) - q19_cash + 100
        score_q19 = MaxPoints_Q19 * (numerator / 200)

        # Ensure score_q19 is within 0 and MaxPoints_Q19
        score_demographic += max(0, min(score_q19, MaxPoints_Q19))

        # Investment Decision-Making (2.03 points)
        decision_points = {
            "Do not have investment assets": 0,
            "Rely on the advice of a professional": 1.85,
            "I make my own investment decisions": 2.03
        }
        score_demographic += decision_points.get(q20_decision_maker, 0)

        # Total Score
        total_score = score_original + score_demographic

        # Risk Aversion Calculation
        S_min = 13  # Minimum possible total score
        S_max = 67  # Updated maximum possible total score
        A_min = 1   # Lowest risk aversion coefficient
        A_max = 10  # Highest risk aversion coefficient

        proportion = (total_score - S_min) / (S_max - S_min)
        risk_aversion = A_max - proportion * (A_max - A_min)

        # Categorize Risk Tolerance into 5 Levels
        categories = ["Very Low Risk Tolerance", "Low Risk Tolerance", "Moderate Risk Tolerance", "High Risk Tolerance", "Very High Risk Tolerance"]
        category_thresholds = [
            S_min + (S_max - S_min) * 0.2,  # 20%
            S_min + (S_max - S_min) * 0.4,  # 40%
            S_min + (S_max - S_min) * 0.6,  # 60%
            S_min + (S_max - S_min) * 0.8   # 80%
        ]

        if total_score <= category_thresholds[0]:
            risk_category = categories[0]
        elif total_score <= category_thresholds[1]:
            risk_category = categories[1]
        elif total_score <= category_thresholds[2]:
            risk_category = categories[2]
        elif total_score <= category_thresholds[3]:
            risk_category = categories[3]
        else:
            risk_category = categories[4]

        # Display Results and Explanations
        st.write("## Quiz Results")
        st.write(f"Your total score is: **{round(total_score, 2)}** out of {S_max}")
        st.write(f"Your estimated risk aversion coefficient is: **{round(risk_aversion, 2)}**")
        st.write(f"Your risk tolerance category is: **{risk_category}**")

        st.write("### Explanation")
        st.write("""
        The risk aversion coefficient is calculated based on your total score from the quiz. The quiz assesses your willingness to take financial risks using the Grable & Lytton method, which considers various factors such as your financial attitudes, behaviors, demographics, and actual investment allocations.

        Each question in the quiz is scored according to its statistical impact on risk tolerance, derived from empirical research findings. For example:

        - **Gender**: Males tend to have higher risk tolerance scores than females, so points are allocated accordingly.
        - **Age**: Certain age groups have higher average risk tolerance scores.
        - **Investment Allocation**: A higher percentage of equities and other assets in your portfolio indicates a higher risk tolerance.

        Your total score is mapped to a risk aversion coefficient between 1 (lowest risk aversion, highest risk tolerance) and 10 (highest risk aversion, lowest risk tolerance). This mapping is done using a normalization process that considers the minimum and maximum possible scores.

        ### How Your Risk Aversion Coefficient is Used
        In the **mean-variance optimization** framework, your risk aversion coefficient helps determine the optimal asset allocation that balances expected returns against portfolio risk (volatility). Specifically, the coefficient is used in the utility function:

        \n
        $$ U = E(R_p) - \\frac{1}{2} \\times A \\times \\sigma_p^2 $$
        \n

        Where:
        - \( U \): Utility of the portfolio
        - \( E(R_p) \): Expected return of the portfolio
        - \( A \): Your risk aversion coefficient
        - \( \sigma_p^2 \): Variance of the portfolio returns

        A higher risk aversion coefficient \( A \) means you prefer less risk, so the optimization process will favor a portfolio with lower volatility (risk), even if it has lower expected returns. Conversely, a lower \( A \) indicates a higher tolerance for risk, leading to a portfolio with potentially higher returns and higher volatility.

        ### Other Objectives Available
        If you wish to explore alternative investment strategies or objectives, you can consider:

        - **Maximizing Sharpe Ratio**: Focusing on portfolios that offer the best risk-adjusted returns.
        - **Minimizing Variance**: Prioritizing the lowest possible portfolio risk regardless of returns.
        - **Custom Constraints**: Incorporating specific asset constraints or investment preferences into the optimization process.

        These objectives can be tailored based on your risk tolerance and investment goals.

        ### Next Steps
        Based on your risk tolerance category and risk aversion coefficient, we can proceed to design an investment portfolio that aligns with your preferences. The mean-variance optimization model will utilize your risk aversion coefficient to recommend an optimal asset allocation.

        Feel free to explore the data visualization page to see how different portfolios perform and to adjust your preferences as needed.
        """)

        # Store the risk aversion in session state
        st.session_state["risk_aversion"] = risk_aversion
        st.session_state["risk_category"] = risk_category

    if st.button("View Data Visualization"):
        # Navigate to Data Visualization page
        st.session_state["current_page"] = "Data Visualization"
        st.rerun()

    else:
        st.stop()


# -------------------------------
# 2. Data Visualization Page
# -------------------------------


def data_visualization_page():
    st.title("Data Visualization")
    st.write("You can explore the dataset before proceeding to optimization.")

    # Option to proceed to Optimization
    if st.button("Proceed to Optimization"):
        st.session_state["current_page"] = "Optimization"
        st.rerun()

    # Visualization code goes here
    visualize_dataset()


def visualize_dataset(top_n_countries=10):
    """
    Visualize dataset with descriptive statistics and meaningful interactive graphs.
    
    Parameters:
    - data (pd.DataFrame): The dataset to visualize.
    - top_n_countries (int): Number of top countries to display in the country distribution plot.
    """
    # Always start with the full data
    data_to_visualize = data.copy()
    returns = data_to_visualize.pct_change().dropna()

    """
    Visualize dataset with descriptive statistics and meaningful interactive graphs.
    
    Parameters:
    - data (pd.DataFrame): The dataset to visualize (filtered or unfiltered).
                           Columns should be ISINs representing different stocks.
    - static_data (pd.DataFrame): Static metadata for stocks, containing at least
                                  'ISIN', 'Region', 'GICSSectorName', and 'Country' columns.
    - top_n_countries (int): Number of top countries to display in the country distribution plot.
    """
    # Ensure 'ISIN' is in static_data
    required_columns = {'ISIN', 'Region', 'GICSSectorName', 'Country'}
    if not required_columns.issubset(static_data.columns):
        st.error(f"static_data must contain the following columns: {required_columns}")
        return
    
    # Extract list of ISINs from the data
    data_isins = data_to_visualize.columns.tolist()
    
    # Filter static_data to include only ISINs present in data
    static_filtered = static_data[static_data['ISIN'].isin(data_isins)]
    
    if static_filtered.empty:
        st.warning("No matching ISINs found in static_data for the filtered dataset.")
        return
    
    # Map ISIN to Company Name
    isin_to_company = static_filtered.set_index('ISIN')['Company'].to_dict()

    # Rename the columns of data_to_visualize from ISINs to Company names
    data_to_visualize = data_to_visualize.rename(columns=isin_to_company)

    returns = returns.rename(columns=isin_to_company)

    # Update static_filtered to include only selected companies
    # Remove the incorrect assignment that caused the KeyError
    # static_filtered['Company'] = data_to_visualize.columns  # Removed

    # Ensure that 'Company' column in static_filtered matches the renamed data_to_visualize
    static_filtered = static_filtered[static_filtered['Company'].isin(data_to_visualize.columns)]

    
    # # Calculate descriptive statistics
    # st.subheader("Descriptive Statistics")
    # st.dataframe(data_to_visualize.describe())
    
    
    # # Correlation Matrix and Interactive Heatmap
    # st.subheader("Correlation Matrix")
    # corr_matrix = data_to_visualize.corr()
    # st.dataframe(corr_matrix)
    
    # st.subheader("Correlation Heatmap")
    # fig_corr = px.imshow(corr_matrix,
    #                      text_auto=True,
    #                      color_continuous_scale='RdBu',
    #                      title='Correlation Heatmap',
    #                      labels=dict(color="Correlation"))
    # fig_corr.update_layout(width=800, height=600)
    # st.plotly_chart(fig_corr, use_container_width=True)
    
    # Create tabs for categorical distributions
    tab1, tab2, tab3 = st.tabs(["By Region", "By Sector", "By Country"])
    
    with tab1:
        st.subheader("Percentage of Stocks by Region")
        region_counts = static_filtered['Region'].value_counts(normalize=True) * 100
        region_df = region_counts.reset_index()
        region_df.columns = ['Region', 'Percentage']
        
        fig_region = px.bar(region_df, 
                            x='Region', 
                            y='Percentage',
                            text='Percentage',
                            title='Distribution of Stocks by Region',
                            labels={'Percentage': 'Percentage (%)'},
                            color='Percentage',
                            color_continuous_scale='viridis')
        fig_region.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_region.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                                 xaxis_title="Region",
                                 yaxis_title="Percentage (%)",
                                 showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with tab2:
        st.subheader("Percentage of Stocks by Sector")
        sector_counts = static_filtered['GICSSectorName'].value_counts(normalize=True) * 100
        sector_df = sector_counts.reset_index()
        sector_df.columns = ['Sector', 'Percentage']
        
        fig_sector = px.bar(sector_df, 
                            x='Sector', 
                            y='Percentage',
                            text='Percentage',
                            title='Distribution of Stocks by Sector',
                            labels={'Percentage': 'Percentage (%)'},
                            color='Percentage',
                            color_continuous_scale='magma')
        fig_sector.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_sector.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                                 xaxis_title="Sector",
                                 yaxis_title="Percentage (%)",
                                 showlegend=False)
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab3:
        st.subheader(f"Percentage of Stocks by Country (Top {top_n_countries})")
        country_counts = static_filtered['Country'].value_counts(normalize=True) * 100
        if len(country_counts) > top_n_countries:
            top_countries = country_counts.head(top_n_countries)
            other_percentage = 100 - top_countries.sum()
            top_countries = pd.concat([top_countries, pd.Series({'Other': other_percentage})])
        else:
            top_countries = country_counts
        country_df = top_countries.reset_index()
        country_df.columns = ['Country', 'Percentage']
        
        fig_country = px.bar(country_df, 
                             x='Country', 
                             y='Percentage',
                             text='Percentage',
                             title=f'Distribution of Stocks by Country (Top {top_n_countries})',
                             labels={'Percentage': 'Percentage (%)'},
                             color='Percentage',
                             color_continuous_scale='RdBu')
        fig_country.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_country.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                                  xaxis_title="Country",
                                  yaxis_title="Percentage (%)",
                                  showlegend=False)
        st.plotly_chart(fig_country, use_container_width=True)

    # ---------------------------
    # Interactive Selection Steps
    # ---------------------------

    st.header("Select Companies for Additional Insights")

    # Step 1: Select Region(s)
    available_regions = static_filtered['Region'].dropna().unique().tolist()
    selected_regions = st.selectbox(
        "Select a Region",
        options=available_regions,
        index=0
    )

    # Step 2: Select Country(s) within Selected Region
    available_countries = static_filtered[static_filtered['Region'] == selected_regions]['Country'].dropna().unique().tolist()
    selected_countries = st.multiselect(
        "Select Country/Countries",
        options=available_countries,
    )

    # Step 3: Select Company(s) within Selected Country(s)
    if selected_countries:
        available_companies = static_filtered[
            (static_filtered['Region'] == selected_regions) &
            (static_filtered['Country'].isin(selected_countries))
        ]['Company'].tolist()
    else:
        available_companies = []

    selected_companies = st.multiselect(
        "Select Company/Companies",
        options=available_companies,
    )

    if selected_companies:
        # Verify that selected_companies are in data_to_visualize.columns
        missing_companies = set(selected_companies) - set(data_to_visualize.columns)
        if missing_companies:
            st.error(f"The following selected companies are not present in the data: {', '.join(missing_companies)}")
            st.stop()

        # Filter data_to_visualize to include only selected companies
        data_to_visualize = data_to_visualize[selected_companies]
        returns = returns[selected_companies]
        
        # Also filter static_filtered accordingly
        static_filtered = static_filtered[static_filtered['Company'].isin(selected_companies)]
        
        # Update session state
        st.session_state["filtered_data"] = data_to_visualize
    else:
        st.warning("No companies selected. Please select at least one company to view insights.")
        st.stop()
    
    # Additional Visualizations
    st.subheader("Additional Insights")

    # Distribution of Numerical Variables - Interactive Histograms
    st.markdown("### Distribution of Returns")
    numeric_columns = data_to_visualize.select_dtypes(include=np.number).columns.tolist()
    numeric_columns_return = data_to_visualize.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_columns_return:
        fig_hist = px.histogram(
            returns,
            x=col,
            nbins=30,
            title=f"Histogram of {col}",
            labels={col: col},
            opacity=0.75,
            marginal="box",
            color_discrete_sequence=['skyblue']
        )
        fig_hist.update_layout(
            showlegend=False,
            xaxis_title=col,
            yaxis_title="Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Box Plots of Numerical Variables - Interactive
    st.markdown("### Box Plots of Returns")
    for col in numeric_columns_return:
        fig_box = px.box(
            returns,
            y=col,
            points="all",
            title=f"Box Plot of {col}",
            labels={col: col},
            color_discrete_sequence=['lightgreen']
        )
        fig_box.update_layout(
            showlegend=False,
            yaxis_title=col
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Pairwise Relationships - Interactive Scatter Matrix
    st.markdown("### Pairwise Scatter Plots with Plotly Scatter Matrix")
    if len(numeric_columns) >= 2:
        # Since 'color' expects a value per observation (row), and sectors are per company (column),
        # it's not appropriate to use 'color' in scatter_matrix in this context.
        # Therefore, we'll omit the 'color' parameter.
        scatter_matrix = px.scatter_matrix(
            returns,
            dimensions=numeric_columns_return,
            title="Scatter Matrix of Returns",
            labels={col: col for col in numeric_columns},
            height=800
            # Removed 'color' parameter to prevent length mismatch
        )
        scatter_matrix.update_layout(showlegend=False)
        st.plotly_chart(scatter_matrix, use_container_width=True)

    # Time-Series Trends - Interactive Line Charts (If Applicable)
    # Assuming the DataFrame index is datetime or there's a 'Date' column
    date_col = None
    # if isinstance(data_to_visualize.index, pd.DatetimeIndex):
    date_col = data_to_visualize.index
    date_col_return = returns.index
    # elif 'Date' in data_to_visualize.columns:
    #     try:
    #         date_col = pd.to_datetime(data_to_visualize['Date'])
    #         data_to_visualize = data_to_visualize.set_index('Date')
    #     except Exception as e:
    #         st.warning("Failed to parse 'Date' column as datetime.")

    if date_col is not None:
        st.markdown("### Time-Series Trends of Prices")
        for col in numeric_columns:
            fig_line = px.line(
                data_to_visualize,
                x=date_col,
                y=col,
                title=f"Trend of {col} Over Time",
                labels={'x': 'Date', col: col},
                markers=True
            )
            fig_line.update_layout(
                showlegend=False,
                xaxis_title="Date",
                yaxis_title=col
            )
            st.plotly_chart(fig_line, use_container_width=True)

    if date_col_return is not None:
        st.markdown("### Time-Series Trends of Returns")
        for col in numeric_columns_return:
            fig_line = px.line(
                returns,
                x=date_col_return,
                y=col,
                title=f"Trend of {col} Over Time",
                labels={'x': 'Date', col: col},
                markers=True
            )
            fig_line.update_layout(
                showlegend=False,
                xaxis_title="Date",
                yaxis_title=col
            )
            st.plotly_chart(fig_line, use_container_width=True)


# # Initialize risk aversion
# if "risk_aversion" not in st.session_state:
#     risk_aversion_quiz()
# else:
#     risk_aversion = st.session_state["risk_aversion"]


# -------------------------------
# 3. Optimization Page
# -------------------------------


def optimization_page():

    st.title("Portfolio Optimization")
    global selected_objective

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

    # Choose objective function
    st.header("Choose Optimization Objective")
    objectives = [
        "Maximum Sharpe Ratio Portfolio",
        "Minimum Global Variance Portfolio",
        "Maximum Diversification Portfolio",
        "Equally Weighted Risk Contribution Portfolio",
        "Inverse Volatility Portfolio",
    ]
    selected_objective = st.selectbox("Select an objective function", objectives)

    # After selecting the objective, display specific constraints
    if selected_objective == "Maximum Sharpe Ratio Portfolio":
        display_constraints()
    elif selected_objective == "Minimum Global Variance Portfolio":
        display_constraints()
    elif selected_objective == "Maximum Diversification Portfolio":
        display_constraints()
    elif selected_objective == "Equally Weighted Risk Contribution Portfolio":
        display_constraints()
    elif selected_objective == "Inverse Volatility Portfolio":
        display_constraints()

    # Get current parameters
    current_params = get_current_params()
    previous_params = st.session_state.get("previous_params", None)

    # Compare current and previous parameters
    if previous_params is not None and current_params != previous_params:
        st.session_state["optimization_run"] = False

    # Update previous parameters
    st.session_state["previous_params"] = current_params

    # Apply filtering
    data_filtered = filter_stocks(
        data,
        regions=selected_regions,
        sectors=selected_sectors,
        countries=selected_countries,
    )

    st.session_state["filtered_data"] = data_filtered

    # Assets list after filtering
    assets = data_filtered.columns.tolist()

    # Run optimization when ready
    if st.button("Run Optimization"):
        run_optimization(selected_objective)
    else:
        st.write('Click "Run Optimization" to compute the optimized portfolio.')

    # Compute and show the efficient frontier once the optimization of the selected objective is done
    if st.session_state["optimization_run"] == True:
        if st.button("Show Efficient Frontier"):
            st.session_state["current_page"] = "Efficient Frontier"
            st.rerun()
        else:
            st.write('Click "Show Efficient Frontier" to display the graph.')

    # Option to view filtered data visualization
    if st.button("View Filtered Data Visualization"):
        st.session_state["current_page"] = "Data Visualization"
        st.rerun()

    # Option to review the quiz
    if st.button("Return to Quiz"):
        st.session_state["current_page"] = "Quiz"
        st.rerun() 


# -------------------------------
# 4. Efficient Frontier Page
# -------------------------------

def efficient_frontier_page():
    st.title("Efficient Frontier")

    # Initialize session state variables
    if "efficient_frontier_run" not in st.session_state:
        st.session_state["efficient_frontier_run"] = False
    
    # Retrieve necessary variables from session state
    weights = st.session_state["weights"]
    mean_returns = st.session_state["mean_returns"]
    cov_matrix = st.session_state["cov_matrix"]
    optimized_returns = st.session_state["optimized_returns"]
    optimized_volatility = st.session_state["optimized_volatility"]
    risk_free_rate = st.session_state["risk_free_rate"]
    include_risk_free_asset = st.session_state["include_risk_free_asset"]
    long_only = st.session_state["long_only"]
    leverage_limit = st.session_state["leverage_limit"]
    leverage_limit_value = st.session_state["leverage_limit_value"]
    min_weight_value = st.session_state["min_weight_value"]
    max_weight_value = st.session_state["max_weight_value"]

    num_assets = len(mean_returns)
    weights_optimal = weights.values

    plot_efficient_frontier(
        mean_returns,
        cov_matrix,
        risk_free_rate,
        include_risk_free_asset,
        weights_optimal,
        long_only,
        leverage_limit,
        leverage_limit_value,
        min_weight_value,
        max_weight_value,
    )
    
    st.session_state["efficient_frontier_run"] = True

    # Option to proceed to Optimization
    if st.button("Return to Optimization"):
        st.session_state["efficient_frontier_run"] = False
        st.session_state["current_page"] = "Optimization"
        st.rerun()

    # Option to view filtered data visualization
    if st.button("View Filtered Data Visualization"):
        st.session_state["efficient_frontier_run"] = False
        st.session_state["current_page"] = "Data Visualization"
        st.rerun()

    # Option to review the quiz
    if st.button("Return to Quiz"):
        st.session_state["efficient_frontier_run"] = False
        st.session_state["current_page"] = "Quiz"
        st.rerun() 


# -------------------------------
# 4. Constraints Function
# -------------------------------


def display_constraints():

    global long_only, use_sentiment, region_filter, sectors_filter, country_filter
    global carbon_footprint, min_weight_constraint, max_weight_constraint, leverage_limit
    global selected_sectors, selected_regions, selected_countries
    global leverage_limit_value, min_weight_value, max_weight_value
    global include_risk_free_asset, risk_free_rate

    # Constraints
    st.header("Constraints Selection")
    long_only = st.checkbox("Long only")
    use_sentiment = st.checkbox("Use sentiment data?")
    region_filter = st.checkbox("Region filter")
    sectors_filter = st.checkbox("Sectors filter")
    country_filter = st.checkbox("Country filter")
    carbon_footprint = st.checkbox("Carbon footprint")
    min_weight_constraint = st.checkbox("Minimum weight constraint")
    max_weight_constraint = st.checkbox("Maximum weight constraint")
    leverage_limit = st.checkbox("Leverage limit")

    if selected_objective == "Maximum Sharpe Ratio Portfolio":
        # Risk-Free Asset Inclusion
        st.header("Risk-Free Asset Inclusion")
        include_risk_free_asset = st.checkbox(
            "Include a Risk-Free Asset in the Optimization?", value=True
        )
        if include_risk_free_asset:
            risk_free_rate = st.number_input(
                "Enter the risk-free rate (e.g., 0.01 for 1%)",
                value=0.001,
                min_value=0.00,
                max_value=1.00,
                step=0.001,
                format="%1f"
            )
        else:
            risk_free_rate = 0.00

    else:
        include_risk_free_asset = False
        risk_free_rate = 0

    st.session_state["risk_free_rate"] = risk_free_rate
    st.session_state["include_risk_free_asset"] = include_risk_free_asset

    # Additional inputs
    if sectors_filter:
        sectors = static_data["GICSSectorName"].unique().tolist()
        selected_sectors = st.multiselect("Select sectors to include", sectors)
    else:
        selected_sectors = None

    if region_filter:
        regions = static_data["Region"].unique().tolist()
        selected_regions = st.multiselect("Select regions to include", regions)
    else:
        selected_regions = None

    if country_filter:
        countries = static_data["Country"].unique().tolist()
        selected_countries = st.multiselect("Select countries to include", countries)
    else:
        selected_countries = None

    if leverage_limit:
        leverage_limit_value = st.number_input(
            "Leverage limit", min_value=0.0, value=1.0
        )
    else:
        leverage_limit_value = 1.0

    if min_weight_constraint:
        min_weight_value = (
            st.number_input(
                "Minimum weight (%)",
                min_value=-(leverage_limit_value * 100.0),
                max_value=(leverage_limit_value * 100),
                value=-(leverage_limit_value * 100.0),
            )
            / 100
        )
    else:
        min_weight_value = -1.0 * leverage_limit_value
    if max_weight_constraint:
        max_weight_value = (
            st.number_input(
                "Maximum weight (%)",
                min_value=-(leverage_limit_value * 100.0),
                max_value=(leverage_limit_value * 100.0),
                value=(leverage_limit_value * 100.0),
            )
            / 100
        )
    else:
        max_weight_value = 1.0 * leverage_limit_value


    # Save the constraints
    st.session_state["long_only"] = long_only
    st.session_state["leverage_limit_value"] = leverage_limit_value
    st.session_state["leverage_limit"] = leverage_limit
    st.session_state["min_weight_value"] = min_weight_value
    st.session_state["max_weight_value"] = max_weight_value


# -------------------------------
# 5. Get Current Parameters
# -------------------------------


# Function to get current parameters
def get_current_params():
    params = {
        "long_only": long_only,
        "use_sentiment": use_sentiment,
        "region_filter": region_filter,
        "selected_regions": (
            tuple(sorted(selected_regions)) if selected_regions else None
        ),
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
        "risk_aversion": st.session_state["risk_aversion"],
        "selected_objective": selected_objective,
    }
    return params


# -------------------------------
# 6. Filter Stocks Function
# -------------------------------


# Filtering based on sectors and countries using ISIN numbers
def filter_stocks(data, regions=None, sectors=None, countries=None):
    all_isins = data.columns.tolist()

    if regions is not None:
        companies_regions = static_data[static_data["Region"].isin(regions)]
        regions_isins = companies_regions["ISIN"].tolist()
        all_isins = list(set(all_isins).intersection(set(regions_isins)))
        st.write(f"Total number of stocks after region filtering: {len(all_isins)}")

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
    st.session_state["filtered_data"] = data_filtered
    return data_filtered


# -------------------------------
# 7. Optimization Function
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


# def optimize_sharpe_portfolio(
#     data,
#     long_only,
#     min_weight,
#     max_weight,
#     leverage_limit_value,
#     risk_free_rate,
#     include_risk_free_asset,
#     risk_aversion,
# ):
#     # Calculate returns, mean returns, and covariance matrix
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean() * 12  # Annualized mean returns
#     cov_matrix = returns.cov() * 12  # Annualized covariance matrix

#     # Prepare a result dictionary
#     result = {
#         "weights": None,
#         "mean_returns": mean_returns,
#         "cov_matrix": cov_matrix,
#         "frontier_returns": None,
#         "frontier_volatility": None,
#         "frontier_weights": None,
#         "max_sharpe_weights": None,
#         "max_sharpe_ratio": None,
#         "max_sharpe_returns": None,
#         "max_sharpe_volatility": None,
#         "status": None,
#     }

#     if len(data) / len(cov_matrix) < 2:

#         st.info(f"Len cov matrix : {len(cov_matrix)}")
#         st.info(f"Number observations : {len(data)}")

#         st.info(
#             f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
#         )

#         cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

#         st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

#         # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
#         # cov_matrix = risk_models.fix_nonpositive_semidefinite(
#         #     cov_matrix
#         # )  # Annualized covariance

#     # Adjust covariance matrix to be positive definite
#     cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
#     cov_matrix_adjusted = pd.DataFrame(
#         cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
#     )

#     # Number of assets and asset list
#     num_assets = len(mean_returns)
#     assets = mean_returns.index.tolist()
#     initial_weights = num_assets * [
#         1.0 / num_assets,
#     ]
#     w = cp.Variable(num_assets)
#     total_weight = cp.sum(w)

#     if not leverage_limit:
#         # **Case 1: No Leverage Limit (Sum of weights equals 1)**
#         st.info("Case 1")
#         # Use convex optimization to maximize Sharpe ratio
#         try:
#             # Define weight bounds
#             if long_only:
#                 weight_bounds = (max(min_weight, 0.0), min(max_weight, 1.0))
#             else:
#                 weight_bounds = (max(min_weight, -1.0), min(max_weight, 1.0))

#             # Initialize Efficient Frontier
#             ef = EfficientFrontier(
#                 mean_returns, cov_matrix_adjusted, weight_bounds=weight_bounds
#             )

#             ef.add_constraint(lambda w: cp.sum(w) == 1)

#             if include_risk_free_asset:
#                 ef.max_sharpe(risk_free_rate=risk_free_rate)

#             else:
#                 # Maximize Utility Function: Maximize expected return minus risk aversion times variance
#                 portfolio_return = mean_returns.values @ w
#                 portfolio_variance = cp.quad_form(
#                     w, cov_matrix_adjusted, assume_PSD=True
#                 )
#                 # portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
#                 utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
#                 objective = cp.Maximize(utility)

#                 # Constraints
#                 constraints = []

#                 # Constraints
#                 constraints.append(total_weight >= 1)
#                 constraints.append(total_weight <= leverage_limit_value)

#                 if long_only:
#                     constraints.append(w >= max(min_weight, 0.0))
#                     constraints.append(w <= min(max_weight, leverage_limit_value))
#                 else:
#                     constraints.append(w >= max(min_weight, -leverage_limit_value))
#                     constraints.append(w <= min(max_weight, leverage_limit_value))

#                 # Solve the problem
#                 prob = cp.Problem(objective, constraints)
#                 prob.solve(solver=cp.OSQP)

#                 if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#                     weights = w.value
#                     weights = pd.Series(weights, index=assets)

#                     result.x = weights.values
#                     result.success = True
#                     result.status = "Optimization succeeded"
#                     result.fun = prob.value

#                     return result, mean_returns, cov_matrix_adjusted

#                 else:
#                     st.error(f"Optimization failed. Problem status: {prob.status}")
#                     result.x = None
#                     result.success = False
#                     result.status = (
#                         f"Optimization failed. Problem status: {prob.status}"
#                     )
#                     result.fun = None

#                     return result, mean_returns, cov_matrix_adjusted

#             weights = ef.clean_weights()
#             weights = pd.Series(weights).reindex(assets)

#             result.x = weights.values
#             result.success = True
#             result.status = "Optimization succeeded"
#             result.fun = None  # Can set to portfolio variance if needed

#             return result, mean_returns, cov_matrix_adjusted

#         except Exception as e:
#             st.error(f"Optimization failed: {e}")
#             result.x = None
#             result.success = False
#             result.status = f"Optimization failed: {e}"
#             result.fun = None

#             return result, mean_returns, cov_matrix_adjusted

#     else:
#         if num_assets < 500 and include_risk_free_asset:
#             # **Case 2: Leverage Limit with Less Than 500 Assets**
#             st.info("Case 2")
#             # Use non-convex optimizer from PyPortfolioOpt
#             try:

#                 constraints = [
#                     {"type": "ineq", "fun": lambda x: leverage_limit_value - np.sum(x)},
#                     {"type": "ineq", "fun": lambda x: np.sum(x) - 1},
#                 ]

#                 if long_only:
#                     bounds = tuple(
#                         (max(min_weight, 0.0), min(max_weight, leverage_limit_value))
#                         for _ in range(num_assets)
#                     )
#                 else:
#                     bounds = tuple(
#                         (
#                             max(min_weight, -leverage_limit_value),
#                             min(max_weight, leverage_limit_value),
#                         )
#                         for _ in range(num_assets)
#                     )

#                 # Objective functions
#                 def neg_sharpe_ratio(weights):
#                     portfolio_return = np.sum(mean_returns * weights)
#                     portfolio_volatility = np.sqrt(
#                         np.dot(weights.T, np.dot(cov_matrix, weights))
#                     )
#                     sharpe_ratio = (
#                         portfolio_return - risk_free_rate
#                     ) / portfolio_volatility
#                     return -sharpe_ratio

#                 def negative_utility(weights):
#                     portfolio_return = np.sum(mean_returns * weights)
#                     portfolio_volatility = np.sqrt(
#                         np.dot(weights.T, np.dot(cov_matrix, weights))
#                     )
#                     utility = portfolio_return - 0.5 * risk_aversion * (
#                         portfolio_volatility**2
#                     )
#                     return -utility

#                 # Choose the appropriate objective function
#                 if include_risk_free_asset:
#                     objective_function = neg_sharpe_ratio
#                 else:
#                     objective_function = negative_utility

#                 # Progress bar
#                 progress_bar = st.progress(0)
#                 iteration_container = st.empty()

#                 max_iterations = 1000  # Set maximum number of iterations for estimation if taking too long

#                 iteration_counter = {"n_iter": 0}

#                 # Callback function to update progress
#                 def callbackF(xk):
#                     iteration_counter["n_iter"] += 1
#                     progress = iteration_counter["n_iter"] / max_iterations
#                     progress_bar.progress(min(progress, 1.0))
#                     iteration_container.text(
#                         f"Iteration: {iteration_counter['n_iter']}"
#                     )

#                 # Estimated time indicator
#                 st.info(
#                     "Estimated time to complete optimization: depends on data and constraints."
#                 )

#                 with st.spinner("Optimization in progress..."):
#                     start_time = time.time()
#                     result = minimize(
#                         objective_function,
#                         initial_weights,
#                         method="SLSQP",
#                         bounds=bounds,
#                         constraints=constraints,
#                         options={"maxiter": max_iterations},
#                         callback=callbackF,
#                     )
#                     end_time = time.time()
#                     elapsed_time = end_time - start_time

#                 progress_bar.empty()
#                 iteration_container.empty()

#                 st.success(f"Optimization completed in {elapsed_time:.2f} seconds")
#                 return result, mean_returns, cov_matrix

#             except Exception as e:
#                 st.error(f"Optimization failed: {e}")
#                 result.x = None
#                 result.success = False
#                 result.status = f"Optimization failed: {e}"
#                 result.fun = None

#                 return result, mean_returns, cov_matrix_adjusted

#         else:
#             # **Case 3: Leverage Limit with 700 or More Assets and risk free asset included**
#             st.info("Case 3")
#             # Use convex approximation (Second-Order Cone Programming)
#             try:
#                 w = cp.Variable(num_assets)
#                 t = cp.Variable()
#                 total_weight = cp.sum(w)

#                 portfolio_return = mean_returns.values @ w
#                 portfolio_variance = cp.quad_form(w, cov_matrix_adjusted)

#                 if include_risk_free_asset:
#                     # Calculate the efficient frontier with updated constraints
#                     frontier_volatility, frontier_returns, frontier_weights = (
#                         calculate_efficient_frontier_qp(
#                             mean_returns,
#                             cov_matrix,
#                             long_only,
#                             leverage_limit_value,
#                             min_weight_value,
#                             max_weight_value,
#                             None,
#                         )
#                     )

#                     # Compute Sharpe Ratios
#                     sharpe_ratios = (
#                         np.array(frontier_returns) - risk_free_rate
#                     ) / np.array(frontier_volatility).flatten()

#                     # Find the maximum Sharpe Ratio
#                     max_sharpe_idx = np.argmax(sharpe_ratios)
#                     max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]
#                     max_sharpe_return = frontier_returns[max_sharpe_idx]
#                     max_sharpe_volatility = frontier_volatility[max_sharpe_idx]
#                     max_sharpe_weights = frontier_weights[max_sharpe_idx]

#                     st.write(f"Sharpe ratio : {max_sharpe_ratio}")

#                     return (
#                         mean_returns,
#                         cov_matrix_adjusted,
#                         frontier_returns,
#                         frontier_volatility,
#                         frontier_weights,
#                         max_sharpe_weights,
#                         max_sharpe_ratio,
#                         max_sharpe_return,
#                         max_sharpe_volatility,
#                     )

#                 else:
#                     # Maximize Utility Function: Maximize expected return minus risk aversion times variance
#                     portfolio_return = mean_returns.values @ w
#                     portfolio_variance = cp.quad_form(
#                         w, cov_matrix_adjusted, assume_PSD=True
#                     )
#                     # portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
#                     utility = (
#                         portfolio_return - 0.5 * risk_aversion * portfolio_variance
#                     )
#                     objective = cp.Maximize(utility)

#                     # Constraints
#                     constraints = []

#                 # Constraints
#                 constraints.append(total_weight >= 1)
#                 constraints.append(total_weight <= leverage_limit_value)

#                 if long_only:
#                     constraints.append(w >= max(min_weight, 0.0))
#                     constraints.append(w <= min(max_weight, leverage_limit_value))
#                 else:
#                     constraints.append(w >= max(min_weight, -leverage_limit_value))
#                     constraints.append(w <= min(max_weight, leverage_limit_value))

#                 # Solve the problem
#                 prob = cp.Problem(objective, constraints)
#                 prob.solve(solver=cp.SCS, verbose=True)
#                 st.info(prob.status)

#                 if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#                     weights = w.value
#                     weights = pd.Series(weights, index=assets)

#                     result.x = weights.values
#                     result.success = True
#                     result.status = "Optimization succeeded"
#                     result.fun = prob.value

#                     return result, mean_returns, cov_matrix_adjusted

#                 else:
#                     st.error(f"Optimization failed. Problem status: {prob.status}")
#                     result.x = None
#                     result.success = False
#                     result.status = (
#                         f"Optimization failed. Problem status: {prob.status}"
#                     )
#                     result.fun = None

#                     return result, mean_returns, cov_matrix_adjusted

#             except Exception as e:
#                 st.error(f"Optimization failed: {e}")
#                 result.x = None
#                 result.success = False
#                 result.status = f"Optimization failed: {e}"
#                 result.fun = None

#                 return result, mean_returns, cov_matrix_adjusted


def optimize_sharpe_portfolio(
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

    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        # cov_matrix = risk_models.fix_nonpositive_semidefinite(
        #     cov_matrix
        # )  # Annualized covariance

    # Adjust covariance matrix to be positive definite
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )

    # Prepare a result dictionary
    result = {
        "weights": None,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix_adjusted,
        "frontier_returns": None,
        "frontier_volatility": None,
        "frontier_weights": None,
        "max_sharpe_weights": None,
        "max_sharpe_ratio": None,
        "max_sharpe_returns": None,
        "max_sharpe_volatility": None,
        "status": None,
    }

    # Number of assets and asset list
    num_assets = len(mean_returns)
    assets = mean_returns.index.tolist()
    initial_weights = num_assets * [
        1.0 / num_assets,
    ]
    w = cp.Variable(num_assets)
    total_weight = cp.sum(w)

    if not leverage_limit:
        # **Case 1: No Leverage Limit (Sum of weights equals 1)**
        st.info("Case 1")
        # Use convex optimization to maximize Sharpe ratio
        try:
            # Define weight bounds
            if long_only:
                weight_bounds = (max(min_weight, 0.0), min(max_weight, 1.0))
            else:
                weight_bounds = (max(min_weight, -1.0), min(max_weight, 1.0))

            # Initialize Efficient Frontier
            ef = EfficientFrontier(
                mean_returns, cov_matrix_adjusted, weight_bounds=weight_bounds
            )

            ef.add_constraint(lambda w: cp.sum(w) == 1)

            if include_risk_free_asset:
                ef.max_sharpe(risk_free_rate=risk_free_rate)

                weights = ef.clean_weights()
                weights = pd.Series(weights).reindex(assets)

                result["weights"] = weights.values
                result["mean_returns"] = mean_returns
                result["cov_matrix"] = cov_matrix_adjusted
                result["status"] = "success"

            else:
                # Maximize Utility Function: Maximize expected return minus risk aversion times variance
                portfolio_return = mean_returns.values @ w
                portfolio_variance = cp.quad_form(
                    w, cov_matrix_adjusted, assume_PSD=True
                )
                # portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
                utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                objective = cp.Maximize(utility)

                # Constraints
                constraints = []

                # Constraints
                constraints.append(total_weight >= 1)
                constraints.append(total_weight <= leverage_limit_value)

                if long_only:
                    constraints.append(w >= max(min_weight, 0.0))
                    constraints.append(w <= min(max_weight, leverage_limit_value))
                else:
                    constraints.append(w >= max(min_weight, -leverage_limit_value))
                    constraints.append(w <= min(max_weight, leverage_limit_value))

                # Solve the problem
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.SCS)

                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    weights = w.value
                    weights = pd.Series(weights, index=assets)

                    result["weights"] = weights.values
                    result["mean_returns"] = mean_returns
                    result["cov_matrix"] = cov_matrix_adjusted
                    result["status"] = "success"

                else:
                    st.error(f"Optimization failed: {e}")
                    result["status"] = "failure"

        except Exception as e:
            st.error(f"Optimization failed: {e}")
            result["status"] = "failure"

    else:
        if num_assets < 500 and include_risk_free_asset:
            # **Case 2: Leverage Limit with Less Than 500 Assets**
            st.info("Case 2")
            # Use non-convex optimizer from PyPortfolioOpt
            try:

                constraints = [
                    {"type": "ineq", "fun": lambda x: leverage_limit_value - np.sum(x)},
                    {"type": "ineq", "fun": lambda x: np.sum(x) - 1},
                ]

                if long_only:
                    bounds = tuple(
                        (max(min_weight, 0.0), min(max_weight, leverage_limit_value))
                        for _ in range(num_assets)
                    )
                else:
                    bounds = tuple(
                        (
                            max(min_weight, -leverage_limit_value),
                            min(max_weight, leverage_limit_value),
                        )
                        for _ in range(num_assets)
                    )

                # Objective functions
                def neg_sharpe_ratio(weights):
                    portfolio_return = np.sum(mean_returns * weights)
                    portfolio_volatility = np.sqrt(
                        np.dot(weights.T, np.dot(cov_matrix_adjusted, weights))
                    )
                    sharpe_ratio = (
                        portfolio_return - risk_free_rate
                    ) / portfolio_volatility
                    return -sharpe_ratio

                def negative_utility(weights):
                    portfolio_return = np.sum(mean_returns * weights)
                    portfolio_volatility = np.sqrt(
                        np.dot(weights.T, np.dot(cov_matrix_adjusted, weights))
                    )
                    utility = portfolio_return - 0.5 * risk_aversion * (
                        portfolio_volatility**2
                    )
                    return -utility

                # Choose the appropriate objective function
                if include_risk_free_asset:
                    objective_function = neg_sharpe_ratio
                else:
                    objective_function = negative_utility

                # Progress bar
                progress_bar = st.progress(0)
                iteration_container = st.empty()

                max_iterations = 1000  # Set maximum number of iterations for estimation if taking too long

                iteration_counter = {"n_iter": 0}

                # Callback function to update progress
                def callbackF(xk):
                    iteration_counter["n_iter"] += 1
                    progress = iteration_counter["n_iter"] / max_iterations
                    progress_bar.progress(min(progress, 1.0))
                    iteration_container.text(
                        f"Iteration: {iteration_counter['n_iter']}"
                    )

                # Estimated time indicator
                st.info(
                    "Estimated time to complete optimization: depends on data and constraints."
                )

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


                weights = pd.Series(result.x, index=assets)
                result["weights"] = weights.values
                result["mean_returns"] = mean_returns
                result["cov_matrix"] = cov_matrix_adjusted
                result["status"] = "success"

                st.success(f"Optimization completed in {elapsed_time:.2f} seconds")

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                result["status"] = "failure"

        else:
            # **Case 3: Leverage Limit with 700 or More Assets and risk free asset included**
            st.info("Case 3")
            # Use convex approximation (Second-Order Cone Programming)
            w = cp.Variable(num_assets)
            t = cp.Variable()
            total_weight = cp.sum(w)

            portfolio_return = mean_returns.values @ w
            portfolio_variance = cp.quad_form(w, cov_matrix_adjusted, assume_PSD=True)

            constraints = []

            # Constraints
            constraints.append(total_weight >= 1)
            constraints.append(total_weight <= leverage_limit_value)

            if long_only:
                constraints.append(w >= max(min_weight, 0.0))
                constraints.append(w <= min(max_weight, leverage_limit_value))
            else:
                constraints.append(w >= max(min_weight, -leverage_limit_value))
                constraints.append(w <= min(max_weight, leverage_limit_value))

            if include_risk_free_asset:

                try:
                    # Calculate the efficient frontier with updated constraints
                    frontier_volatility, frontier_returns, frontier_weights = (
                        calculate_efficient_frontier_qp(
                            mean_returns,
                            cov_matrix_adjusted,
                            long_only,
                            leverage_limit,
                            leverage_limit_value,
                            min_weight_value,
                            max_weight_value,
                            None,
                        )
                    )

                    # Compute Sharpe Ratios
                    sharpe_ratios = (
                        np.array(frontier_returns) - risk_free_rate
                    ) / np.array(frontier_volatility).flatten()

                    # Find the maximum Sharpe Ratio
                    max_sharpe_idx = np.argmax(sharpe_ratios)
                    max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]
                    max_sharpe_return = frontier_returns[max_sharpe_idx]
                    max_sharpe_volatility = frontier_volatility[max_sharpe_idx]
                    max_sharpe_weights = frontier_weights[max_sharpe_idx]

                    result["mean_returns"] = mean_returns
                    result["cov_matrix"] = cov_matrix_adjusted
                    result["frontier_returns"] = frontier_returns
                    result["frontier_volatility"] = frontier_volatility
                    result["frontier_weights"] = frontier_weights
                    result["max_sharpe_weights"] = max_sharpe_weights
                    result["max_sharpe_ratio"] = max_sharpe_ratio
                    result["max_sharpe_returns"] = max_sharpe_return
                    result["max_sharpe_volatility"] = max_sharpe_volatility
                    result["status"] = "success"

                except Exception as e:
                    st.error(f"Optimization risk free failed: {e}")
                    result["status"] = "failure"

            else:
                # Maximize Utility Function: Maximize expected return minus risk aversion times variance
                portfolio_return = mean_returns.values @ w
                portfolio_variance = cp.quad_form(
                    w, cov_matrix_adjusted, assume_PSD=True
                )
                # portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
                utility = (
                    portfolio_return - 0.5 * risk_aversion * portfolio_variance
                )
                objective = cp.Maximize(utility)


                # Solve the problem
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.SCS, verbose=True)
                st.info(prob.status)

                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    weights = w.value
                    weights = pd.Series(weights, index=assets)

                    result["weights"] = weights.values
                    result["mean_returns"] = mean_returns
                    result["cov_matrix"] = cov_matrix_adjusted
                    result["status"] = "success"

                else:
                    st.error(f"Optimization failed: {e}")
                    result["status"] = "failure"

    return result


# def optimize_sharpe_portfolio(
#     data,
#     long_only,
#     min_weight,
#     max_weight,
#     leverage_limit_value,
#     risk_free_rate,
#     include_risk_free_asset,
#     risk_aversion,
# ):
#     # Calculate returns, mean returns, and covariance matrix
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean() * 12  # Annualized mean returns
#     cov_matrix = returns.cov() * 12  # Annualized covariance matrix

#     # Prepare a result dictionary
#     result = {
#         "weights": None,
#         "mean_returns": mean_returns,
#         "cov_matrix": cov_matrix,
#         "frontier_returns": None,
#         "frontier_volatility": None,
#         "frontier_weights": None,
#         "max_sharpe_weights": None,
#         "max_sharpe_ratio": None,
#         "max_sharpe_returns": None,
#         "max_sharpe_volatility": None,
#         "status": None,
#     }

#     num_assets = len(mean_returns)
#     assets = mean_returns.index.tolist()

#     # Adjust covariance matrix
#     cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
#     cov_matrix_adjusted = pd.DataFrame(
#         cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
#     )

#     if leverage_limit_value and num_assets >= 500 and include_risk_free_asset:
#         # Case when leverage limit, large number of assets, and risk-free asset included
#         st.info("Case: Leverage limit with large dataset and risk-free asset")

#         try:
#             # Compute efficient frontier
#             frontier_volatility, frontier_returns, frontier_weights = (
#                 calculate_efficient_frontier_qp(
#                     mean_returns,
#                     cov_matrix_adjusted,
#                     long_only,
#                     leverage_limit_value,
#                     min_weight,
#                     max_weight,
#                     None,
#                 )
#             )

#             # Compute Sharpe Ratios
#             sharpe_ratios = (np.array(frontier_returns) - risk_free_rate) / np.array(
#                 frontier_volatility
#             ).flatten()

#             # Find the maximum Sharpe Ratio
#             max_sharpe_idx = np.argmax(sharpe_ratios)
#             max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]
#             max_sharpe_return = frontier_returns[max_sharpe_idx]
#             max_sharpe_volatility = frontier_volatility[max_sharpe_idx]
#             max_sharpe_weights = frontier_weights[max_sharpe_idx]

#             result["mean_returns"] = mean_returns
#             result["cov_matrix"] = cov_matrix_adjusted
#             result["frontier_returns"] = frontier_returns
#             result["frontier_volatility"] = frontier_volatility
#             result["frontier_weights"] = frontier_weights
#             result["max_sharpe_weights"] = max_sharpe_weights.values
#             result["max_sharpe_ratio"] = max_sharpe_ratio
#             result["max_sharpe_returns"] = max_sharpe_return
#             result["max_sharpe_volatility"] = max_sharpe_volatility
#             result["status"] = "success"

#         except Exception as e:
#             st.error(f"Optimization failed: {e}")
#             result["status"] = "failure"

#     else:
#         # Other cases
#         st.info("Case: Standard Sharpe Ratio Optimization")
#         try:
#             # Define weight bounds
#             if long_only:
#                 weight_bounds = (max(min_weight, 0.0), min(max_weight, 1.0))
#             else:
#                 weight_bounds = (max(min_weight, -1.0), min(max_weight, 1.0))

#             # Initialize Efficient Frontier
#             ef = EfficientFrontier(
#                 mean_returns, cov_matrix_adjusted, weight_bounds=weight_bounds
#             )

#             if include_risk_free_asset:
#                 ef.max_sharpe(risk_free_rate=risk_free_rate)
#             else:
#                 ef.max_quadratic_utility(risk_aversion=risk_aversion)

#             weights = ef.clean_weights()
#             weights = pd.Series(weights).reindex(assets)

#             result["weights"] = weights.values
#             result["mean_returns"] = mean_returns
#             result["cov_matrix"] = cov_matrix_adjusted
#             result["status"] = "success"

#         except Exception as e:
#             st.error(f"Optimization failed: {e}")
#             result["status"] = "failure"

#     return result


def optimize_min_variance_portfolio(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
):
    # Calculate returns, mean returns, and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12  # Annualized mean returns
    cov_matrix = returns.cov() * 12  # Annualized covariance matrix

    num_assets = len(mean_returns)
    assets = mean_returns.index.tolist()

    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        # cov_matrix = risk_models.fix_nonpositive_semidefinite(
        #     cov_matrix
        # )  # Annualized covariance

    # Adjust covariance matrix
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )

    # Prepare a result dictionary
    result = {
        "weights": None,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix_adjusted,
        "status": None,
    }

    # Define variables
    w = cp.Variable(num_assets)

    # Objective function
    portfolio_variance = cp.quad_form(w, cov_matrix_adjusted, assume_PSD=True)

    # Constraints
    constraints = []

    if leverage_limit_value:
        constraints += [cp.sum(w) == leverage_limit_value]
        if long_only:
            constraints += [
                w >= max(min_weight, 0.0),
                w <= min(max_weight, leverage_limit_value),
            ]
        else:
            constraints += [ 
                w >= max(min_weight, -leverage_limit_value),
                w <= min(max_weight, leverage_limit_value),
            ]

    else:
        constraints += [cp.sum(w) == 1]
        if long_only:
            constraints += [w >= max(min_weight, 0.0), w <= min(max_weight, 1.0)]
        else:
            constraints += [w >= max(min_weight, -1.0), w <= min(max_weight, 1.0)]

    # Formulate the problem
    objective = cp.Minimize(portfolio_variance)

    # Solve the problem
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            weights = pd.Series(weights, index=assets)
            result["weights"] = weights.values
            result["status"] = "success"
        else:
            st.error(f"Optimization failed. Status: {prob.status}")
            result["status"] = "failure"

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        result["status"] = "failure"

    return result


def optimize_max_diversification_portfolio(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
):
    # Calculate returns, mean returns, and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12  # Annualized mean returns
    cov_matrix = returns.cov() * 12  # Annualized covariance matrix

    num_assets = len(mean_returns)
    assets = mean_returns.index.tolist()

    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        # cov_matrix = risk_models.fix_nonpositive_semidefinite(
        #     cov_matrix
        # )  # Annualized covariance

    # Adjust covariance matrix
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )
    std_devs = np.sqrt(np.diag(cov_matrix_adjusted))  # Standard deviations

    # Prepare a result dictionary
    result = {
        "weights": None,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix_adjusted,
        "status": None,
    }

    constraints = []
    bounds = []

    if leverage_limit:
        constraints.append({"type": "ineq", "fun": lambda x: np.sum(x) - 1})
        constraints.append({"type": "ineq", "fun": lambda x: leverage_limit_value - np.sum(x)})
        for _ in range(num_assets):
            if long_only:
                bounds.append((
                    max(min_weight, 0.0),
                    min(max_weight, leverage_limit_value),
                ))
            else:
                bounds.append((
                    max(min_weight, -leverage_limit_value),
                    min(max_weight, leverage_limit_value),
                ))

    else:
        constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        for _ in range(num_assets):
            if long_only:
                bounds.append((max(min_weight, 0.0), min(max_weight, 1.0)))
            else:
                bounds.append((max(min_weight, -1.0), min(max_weight, 1.0)))

    # Initial guess
    x0 = np.full(num_assets, 1.0 / num_assets)

    # Objective function: Negative Diversification Ratio
    def negative_diversification_ratio(w, std_devs, cov_matrix):
        numerator = np.dot(w, std_devs)
        denominator = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        # To avoid division by zero
        if denominator == 0:
            return 1e10
        diversification_ratio = numerator / denominator
        return -diversification_ratio  # Negative for minimization

    # Solve the problem
    try:
        # Progress bar
        progress_bar = st.progress(0)
        iteration_container = st.empty()

        max_iterations = 1000  # Set maximum number of iterations for estimation if taking too long

        iteration_counter = {"n_iter": 0}

        # Callback function to update progress
        def callbackF(xk):
            iteration_counter["n_iter"] += 1
            progress = iteration_counter["n_iter"] / max_iterations
            progress_bar.progress(min(progress, 1.0))
            iteration_container.text(
                f"Iteration: {iteration_counter['n_iter']}"
            )

        # Estimated time indicator
        st.info(
            "Estimated time to complete optimization: depends on data and constraints."
        )

        with st.spinner("Optimization in progress..."):
            start_time = time.time()
            # Optimization
            result_sci = minimize(
                negative_diversification_ratio,
                x0,
                args=(std_devs, cov_matrix_adjusted),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations},
                callback=callbackF,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
        progress_bar.empty()
        iteration_container.empty()

        if result_sci.success:
            weights = result_sci.x
            weights = pd.Series(weights, index=assets)
            result["weights"] = weights.values
            result["mean_returns"] = mean_returns
            result["cov_matrix"] = cov_matrix_adjusted
            result["status"] = "success"
            st.success(f"Optimization completed in {elapsed_time:.2f} seconds")
            # Compute the diversification ratio
            numerator = np.dot(weights, std_devs)
            denominator = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_adjusted, weights)))
            max_diversification_ratio = numerator / denominator
            st.write(f"\nMaximum Diversification Ratio: {max_diversification_ratio:.4f}")

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        result["status"] = "failure"

    return result


def optimize_erc_portfolio(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
):
    # Calculate returns, mean returns, and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12  # Annualized mean returns
    cov_matrix = returns.cov() * 12  # Annualized covariance matrix

    num_assets = len(mean_returns)
    assets = mean_returns.index.tolist()

    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        # cov_matrix = risk_models.fix_nonpositive_semidefinite(
        #     cov_matrix
        # )  # Annualized covariance

    # Adjust covariance matrix
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )

    # Prepare a result dictionary
    result = {
        "weights": None,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix_adjusted,
        "status": None,
    }

    constraints = []
    bounds = []

    if leverage_limit:
        constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - leverage_limit_value})
        for _ in range(num_assets):
            if long_only:
                bounds.append((
                    max(min_weight, 0.0),
                    min(max_weight, leverage_limit_value),
                ))
            else:
                bounds.append((
                    max(min_weight, -leverage_limit_value),
                    min(max_weight, leverage_limit_value),
                ))

    else:
        constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        for _ in range(num_assets):
            if long_only:
                bounds.append((max(min_weight, 0.0), min(max_weight, 1.0)))
            else:
                bounds.append((max(min_weight, -1.0), min(max_weight, 1.0)))

    # Initial guess
    x0 = np.full(num_assets, 1.0 / num_assets)

    # Objective function: Sum of squared differences of risk contributions
    def objective(w, cov_matrix):
        sigma_p = np.sqrt(w.T @ cov_matrix @ w)
        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib
        rc = risk_contrib / sigma_p
        avg_rc = sigma_p / num_assets
        return np.sum((rc - avg_rc)**2)

    # Solve the problem
    try:
        # Progress bar
        progress_bar = st.progress(0)
        iteration_container = st.empty()

        max_iterations = 1000  # Set maximum number of iterations for estimation if taking too long

        iteration_counter = {"n_iter": 0}

        # Callback function to update progress
        def callbackF(xk):
            iteration_counter["n_iter"] += 1
            progress = iteration_counter["n_iter"] / max_iterations
            progress_bar.progress(min(progress, 1.0))
            iteration_container.text(
                f"Iteration: {iteration_counter['n_iter']}"
            )

        # Estimated time indicator
        st.info(
            "Estimated time to complete optimization: depends on data and constraints."
        )

        with st.spinner("Optimization in progress..."):
            start_time = time.time()
            # Optimization
            result_sci = minimize(
                objective,
                x0,
                args=(cov_matrix_adjusted),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations},
                callback=callbackF,
                )
            end_time = time.time()
            elapsed_time = end_time - start_time
        progress_bar.empty()
        iteration_container.empty()

        if result_sci.success:
            weights = result_sci.x
            weights = pd.Series(weights, index=assets)
            result["weights"] = weights.values
            result["mean_returns"] = mean_returns
            result["cov_matrix"] = cov_matrix_adjusted
            result["status"] = "success"
            st.success(f"Optimization completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        result["status"] = "failure"

    return result


def optimize_inverse_volatility_portfolio(
    data,
    min_weight,
    max_weight,
):
    # Calculate returns, mean returns, and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 12  # Annualized mean returns
    cov_matrix = returns.cov() * 12  # Annualized covariance matrix


    if len(data) / len(cov_matrix) < 2:

        st.info(f"Len cov matrix : {len(cov_matrix)}")
        st.info(f"Number observations : {len(data)}")

        st.info(
            f"Ratio of observations / nb. of assets is below 2, current ratio: {len(data) / len(cov_matrix)}. We use shrinkage. "
        )

        cov_matrix = risk_models.CovarianceShrinkage(data, frequency=12).ledoit_wolf()

        st.info("Covariance matrix shrinked using Ledoit_Wolf. ")

        # # Use Ledoit-Wolf shrinkage to ensure the covariance matrix is positive semidefinite
        # cov_matrix = risk_models.fix_nonpositive_semidefinite(
        #     cov_matrix
        # )  # Annualized covariance

    # Adjust covariance matrix
    cov_matrix_adjusted = adjust_covariance_matrix(cov_matrix.values)
    cov_matrix_adjusted = pd.DataFrame(
        cov_matrix_adjusted, index=cov_matrix.index, columns=cov_matrix.columns
    )

    std_devs = np.sqrt(np.diag(cov_matrix_adjusted))  # Annualized standard deviations

    # Prepare a result dictionary
    result = {
        "weights": None,
        "mean_returns": None,
        "cov_matrix": cov_matrix_adjusted,
        "status": None,
    }

    # Inverse of volatilities
    inv_vol = 1 / std_devs
    weights = inv_vol / inv_vol.sum()
    weights = weights * leverage_limit_value

    # Apply constraints
    weights = np.clip(weights, min_weight, max_weight)
    # weights /= weights.sum()

    result["weights"] = weights
    result["mean_returns"] = mean_returns
    result["cov_matrix"] = cov_matrix_adjusted
    result["status"] = "success"

    return result


# -------------------------------
# 8. Efficient Frontier Calculation
# -------------------------------


def calculate_efficient_frontier_qp(
    mean_returns,
    cov_matrix,
    long_only,
    leverage_limit,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
    optimized_returns,
):
    num_assets = len(mean_returns)
    cov_matrix = cov_matrix.values
    mean_returns = mean_returns.values

    # Define variables
    w = cp.Variable(num_assets)
    portfolio_return = mean_returns.T @ w
    portfolio_variance = cp.quad_form(w, cov_matrix, assume_PSD=True)
    assets = st.session_state["filtered_data"].columns.tolist()
    

    if leverage_limit:
        # Leverage limit constraint
        # Constraints
        constraints = [cp.sum(w) >= 1]
        constraints += [cp.sum(w) <= leverage_limit_value]

        if long_only:
            constraints += [
                w >= max(min_weight_value, 0.0),
                w <= min(max_weight_value, leverage_limit_value),
            ]
        else:
            constraints += [
                w >= max(min_weight_value, -leverage_limit_value),
                w <= min(max_weight_value, leverage_limit_value),
            ]
    else:
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints += [
                w >= max(min_weight_value, 0.0),
                w <= min(max_weight_value, 1),
            ]
        else:
            constraints += [
                w >= max(min_weight_value, -1),
                w <= min(max_weight_value, 1),
            ]

    if leverage_limit and len(assets) >= 500 and include_risk_free_asset:
        if long_only:
            # Target returns for the efficient frontier
            target_returns = np.linspace(
                -mean_returns.max(),
                mean_returns.max() * leverage_limit_value * 0.7,
                50,
            )
        else:
            # Target returns for the efficient frontier
            target_returns = np.linspace(
                -mean_returns.max() * leverage_limit_value,
                mean_returns.max() * leverage_limit_value * (1 + risk_free_rate) * 2,
                50,
            )
    else:
        # Target returns for the efficient frontier
        target_returns = np.linspace(
            -mean_returns.max(),
            optimized_returns * 3,
            30,
        )

    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []

    # Prepare result similar to scipy.optimize result
    class Result:
        pass

    result = Result()

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
            vol = np.sqrt(portfolio_variance.value)  # Annualized volatility
            frontier_volatility.append(vol)
            # st.write(f"Frontier volatility : {frontier_volatility}")
            frontier_returns.append(target_return)
            # st.write(f"Frontier return : {frontier_returns}")
            weights = w.value
            weights = pd.Series(weights, index=assets)
            result.x = weights.values
            result.success = True
            result.status = "Optimization succeeded"
            result.fun = prob.value
            frontier_weights.append(result.x)
        else:
            st.warning(
                f"Optimization failed for target return {target_return:.2%}. Status: {prob.status}"
            )
            continue

    return frontier_volatility, frontier_returns, frontier_weights


# -------------------------------
# 8. Run Optimization Function
# -------------------------------


def run_optimization(selected_objective):
    # Retrieve constraints from user inputs
    constraints = {
        "min_weight": min_weight_value,
        "max_weight": max_weight_value,
        "long_only": long_only,
        "leverage_limit": leverage_limit_value,
    }

    # Use filtered data if available
    if "filtered_data" in st.session_state:
        data_to_use = st.session_state["filtered_data"]
    else:
        data_to_use = data

    # Retrieve risk aversion from session state
    if "risk_aversion" in st.session_state:
        risk_aversion = st.session_state["risk_aversion"]
    else:
        risk_aversion = 1  # Default value

    if selected_objective == "Maximum Sharpe Ratio Portfolio":
        result = optimize_sharpe_portfolio(
            data_to_use,
            long_only,
            min_weight_value,
            max_weight_value,
            leverage_limit_value,
            risk_free_rate,
            include_risk_free_asset,
            risk_aversion,
        )
    elif selected_objective == "Minimum Global Variance Portfolio":
        result = optimize_min_variance_portfolio(
            data_to_use,
            long_only,
            constraints["min_weight"],
            constraints["max_weight"],
            constraints["leverage_limit"],
        )
    elif selected_objective == "Maximum Diversification Portfolio":
        result = optimize_max_diversification_portfolio(
            data_to_use,
            long_only,
            constraints["min_weight"],
            constraints["max_weight"],
            constraints["leverage_limit"],
        )
    elif selected_objective == "Equally Weighted Risk Contribution Portfolio":
        result = optimize_erc_portfolio(
            data_to_use,
            long_only,
            constraints["min_weight"],
            constraints["max_weight"],
            constraints["leverage_limit"],
        )
    elif selected_objective == "Inverse Volatility Portfolio":
        result = optimize_inverse_volatility_portfolio(
            data_to_use,
            constraints["min_weight"],
            constraints["max_weight"],
        )
    else:
        st.error("Invalid objective selected.")
        return

    if result["status"] == "success":
        process_optimization_result(result, data_to_use, selected_objective)
    else:
        st.error("Optimization failed.")


# -------------------------------
# 9. Process Optimization Result
# -------------------------------


def process_optimization_result(result, data, selected_objective):
    if result is None or result["status"] != "success":
        st.error("Optimization failed.")
        return
    
    # Retrieve risk aversion from session state
    if 'risk_aversion' in st.session_state:
        risk_aversion = st.session_state['risk_aversion']
    else:
        risk_aversion = 1  # Default value

    # Depending on what is in the result dictionary, process accordingly
    if result["weights"] is not None:
        weights = result["weights"]
    elif result.get("max_sharpe_weights") is not None:
        weights = result["max_sharpe_weights"]
    else:
        st.error("No weights found in the result.")  
        return

    weights = pd.Series(weights, index=data.columns)
    st.session_state["optimization_run"] = True
    st.session_state["weights"] = weights

    # Mean returns and covariance matrix
    mean_returns = result["mean_returns"]
    cov_matrix = result["cov_matrix"]
    st.session_state["mean_returns"] = mean_returns
    st.session_state["cov_matrix"] = cov_matrix

    if result.get("max_sharpe_returns") is not None:
        portfolio_return = result["max_sharpe_returns"]
    elif mean_returns is not None:
        portfolio_return = np.sum(mean_returns * weights)
    else:
        portfolio_return = None

    if result.get("max_sharpe_volatility") is not None:
        portfolio_volatility = result["max_sharpe_volatility"]
    elif cov_matrix is not None:
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    else:
        portfolio_volatility = None

    st.session_state["optimized_returns"] = portfolio_return
    st.session_state["optimized_volatility"] = portfolio_volatility

    if portfolio_return is not None and portfolio_volatility is not None:
        if include_risk_free_asset:
            # Calculate Sharpe Ratio
            if result.get("max_sharpe_ratio") is not None:
                sharpe_ratio = result["max_sharpe_ratio"]
            else:
                sharpe_ratio = (
                    portfolio_return - risk_free_rate
                ) / portfolio_volatility

            st.subheader(f"Portfolio Performance ({selected_objective}):")
            st.write(f"Expected Annual Return: {portfolio_return:.2%}")
            st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            # Calculate allocation between risk-free asset and portfolio
            allocation_tangency = (portfolio_return - risk_free_rate) / (
                risk_aversion * (portfolio_volatility**2)
            )
            allocation_tangency = min(max(allocation_tangency, 0), sum(weights))
            allocation_risk_free = max(sum(weights) - allocation_tangency, 0)

            st.write(f"Invest {allocation_tangency * 100:.2f}% in the portfolio.")
            st.write(
                f"Invest {allocation_risk_free * 100:.2f}% in the risk-free asset."
            )
        else:
            # Calculate Sharpe Ratio without risk-free asset
            sharpe_ratio = portfolio_return / portfolio_volatility

            st.subheader(f"Portfolio Performance ({selected_objective}):")
            st.write(f"Expected Annual Return: {portfolio_return:.2%}")
            st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        st.subheader(f"Portfolio Weights ({selected_objective}):")
        st.write(
            "Cannot compute portfolio return and volatility without returns and covariance matrix."
        )

    # Show the allocation
    allocation_df = pd.DataFrame({"ISIN": data.columns, "Weight": weights})
    # Map ISIN to Company name
    isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))
    allocation_df["Company"] = allocation_df["ISIN"].map(isin_to_company)
    allocation_df = allocation_df[["Company", "Weight"]]

    st.subheader("Portfolio Weights:")
    st.write(allocation_df)
    st.write(f"Sum of the weights: {np.sum(weights)}")


# -------------------------------
# 10. Plotting Functions
# -------------------------------

# def plot_efficient_frontier_with_data(
#     mean_returns,
#     cov_matrix,
#     weights_optimal,
#     frontier_returns,
#     frontier_volatility,
#     max_sharpe_volatility,
#     max_sharpe_returns,
#     risk_free_rate,
#     include_risk_free_asset,
# ):
#     plt.figure(figsize=(10, 7))

#     # Plot Efficient Frontier
#     plt.plot(
#         frontier_volatility,
#         frontier_returns,
#         "r--",
#         linewidth=3,
#         label="Efficient Frontier",
#     )

#     # Plot Individual Assets
#     assets = mean_returns.index.tolist()
#     asset_returns = mean_returns.values
#     asset_volatility = np.sqrt(np.diag(cov_matrix.values))
#     plt.scatter(
#         asset_volatility,
#         asset_returns,
#         marker="o",
#         color="blue",
#         s=10,
#         label="Individual Assets",
#     )

#     if include_risk_free_asset and max_sharpe_volatility is not None:
#         # Plot the Capital Market Line
#         cml_x = [0, max_sharpe_volatility]
#         cml_y = [risk_free_rate, max_sharpe_returns]
#         plt.plot(
#             cml_x, cml_y, color="green", linestyle="--", label="Capital Market Line"
#         )

#         # Highlight the tangency portfolio
#         plt.scatter(
#             max_sharpe_volatility,
#             max_sharpe_returns,
#             marker="*",
#             color="red",
#             s=500,
#             label="Tangency Portfolio",
#         )
#     else:
#         # Highlight the optimal portfolio
#         portfolio_return = np.sum(mean_returns * weights_optimal)
#         portfolio_volatility = np.sqrt(
#             np.dot(weights_optimal.T, np.dot(cov_matrix.values, weights_optimal))
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


# -------------------------------
# 9. Plotting Function
# -------------------------------


# Efficient Frontier Plotting Function
def plot_efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    weights_optimal,
    long_only,
    leverage_limit,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):

    tangency_return = np.sum(mean_returns * weights_optimal)

    # Calculate the efficient frontier with updated constraints
    frontier_volatility, frontier_returns, frontier_weights = (
        calculate_efficient_frontier_qp(
            mean_returns,
            cov_matrix,
            long_only,
            leverage_limit,
            leverage_limit_value,
            min_weight_value,
            max_weight_value,
            tangency_return,
        )
    )

    # Plotting
    plt.figure(figsize=(10, 7))

    # plt.colorbar(label="Sharpe Ratio")
    plt.plot(
        frontier_volatility,
        frontier_returns,
        "r--",
        linewidth=3,
        label="Efficient Frontier",
    )

    # Plot Individual Assets
    assets = mean_returns.index.tolist()
    asset_returns = mean_returns.values
    asset_volatility = np.sqrt(np.diag(cov_matrix.values))
    plt.scatter(
        asset_volatility,
        asset_returns,
        marker="o",
        color="blue",
        s=1,
        label="Individual Assets",
    )

    # # Annotate each asset
    # for i, asset in enumerate(assets):
    #     plt.annotate(
    #         asset,
    #         (asset_volatility[i], asset_returns[i]),
    #         textcoords="offset points",
    #         xytext=(5, 5),
    #         ha="left",
    #     )

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
# Run the App
# -------------------------------

if __name__ == "__main__":
    main()


# # -------------------------------
# # 10. Main Application Logic
# # -------------------------------


# if leverage_limit and len(assets) >= 500 and include_risk_free_asset:
#     if st.button("Run Optimization"):
#         (
#             mean_returns,
#             cov_matrix,
#             frontier_returns,
#             frontier_volatility,
#             frontier_weights,
#             max_sharpe_weights,
#             max_sharpe_ratio,
#             max_sharpe_returns,
#             max_sharpe_volatility,
#         ) = optimize_sharpe_portfolio(
#             data,
#             long_only,
#             min_weight_value,
#             max_weight_value,
#             leverage_limit_value,
#             risk_free_rate,
#             include_risk_free_asset,
#             risk_aversion,
#         )

#         weights = pd.Series(max_sharpe_weights.x, index=assets)

#         # Calculate Sharpe Ratio
#         sharpe_ratio = max_sharpe_ratio

#         # Calculate allocation between risk-free asset and tangency portfolio
#         allocation_tangency = (max_sharpe_returns - risk_free_rate) / (
#             risk_aversion * (max_sharpe_volatility**2)
#         )
#         allocation_tangency = min(max(allocation_tangency, 0), sum(weights))
#         allocation_risk_free = max(sum(weights) - allocation_tangency, 0)

#         st.subheader("Portfolio Performance with Risk-Free Asset:")
#         st.write(f"Expected Annual Return: {max_sharpe_returns:.2%}")
#         st.write(f"Annual Volatility: {max_sharpe_volatility:.2%}")
#         st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
#         st.write(f"Invest {allocation_tangency * 100:.2f}% in the tangency portfolio.")
#         st.write(f"Invest {allocation_risk_free * 100:.2f}% in the risk-free asset.")

#         # Show the allocation
#         allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
#         # Create a mapping of ISIN to Company name from static_data
#         isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))

#         # Replace ISIN in allocation_df with the corresponding company names
#         allocation_df["ISIN"] = allocation_df["ISIN"].map(isin_to_company)

#         # Optionally rename the column to reflect the new data
#         allocation_df.rename(columns={"ISIN": "Company"})

#         st.subheader("Tangency Portfolio Weights:")
#         st.write(allocation_df)
#         st.write(f"Sum of the weights: {np.sum(weights)}")

#         # Plotting
#         plt.figure(figsize=(10, 7))

#         plt.plot(
#             frontier_volatility,
#             frontier_returns,
#             "r--",
#             linewidth=3,
#             label="Efficient Frontier",
#         )

#         # Plot Individual Assets
#         assets = mean_returns.index.tolist()
#         asset_returns = mean_returns.values
#         asset_volatility = np.sqrt(np.diag(cov_matrix.values))
#         plt.scatter(
#             asset_volatility,
#             asset_returns,
#             marker="o",
#             color="blue",
#             s=1,
#             label="Individual Assets",
#         )

#         # Plot the Capital Market Line
#         cml_x = [0, max_sharpe_volatility]
#         cml_y = [risk_free_rate, max_sharpe_returns]
#         plt.plot(
#             cml_x, cml_y, color="green", linestyle="--", label="Capital Market Line"
#         )

#         # Highlight the tangency portfolio
#         plt.scatter(
#             max_sharpe_volatility,
#             max_sharpe_returns,
#             marker="*",
#             color="red",
#             s=500,
#             label="Tangency Portfolio",
#         )

#         plt.title("Efficient Frontier with Random Portfolios")
#         plt.xlabel("Annualized Volatility")
#         plt.ylabel("Annualized Expected Returns")
#         plt.legend()
#         st.pyplot(plt)

#     else:
#         st.write('Click "Run Optimization" to compute the optimized portfolio.')


# else:
#     if st.button("Run Optimization"):
#         tangency_result, tangency_mean_returns, tangency_cov_matrix = (
#             optimize_sharpe_portfolio(
#                 data,
#                 long_only,
#                 min_weight_value,
#                 max_weight_value,
#                 leverage_limit_value,
#                 risk_free_rate,
#                 include_risk_free_asset,
#                 risk_aversion,
#             )
#         )
#         weights = pd.Series(tangency_result.x, index=assets)
#         st.session_state["optimization_run"] = True
#         st.session_state["weights"] = weights
#         st.session_state["mean_returns"] = tangency_mean_returns
#         st.session_state["cov_matrix"] = tangency_cov_matrix

#         # Display optimization results
#         # st.write(weights.apply(lambda x: f"{x:.2%}"))

#         # Calculate portfolio performance
#         portfolio_return = np.sum(tangency_mean_returns * weights)  # Annualized return
#         portfolio_volatility = np.sqrt(
#             np.dot(weights.T, np.dot(tangency_cov_matrix, weights))
#         )  # Annualized volatility

#         st.session_state["optimized_returns"] = portfolio_return
#         st.session_state["optimized_volatility"] = portfolio_volatility

#         if include_risk_free_asset:
#             # Calculate Sharpe Ratio
#             sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

#             # Calculate allocation between risk-free asset and tangency portfolio
#             allocation_tangency = (portfolio_return - risk_free_rate) / (
#                 risk_aversion * (portfolio_volatility**2)
#             )
#             allocation_tangency = min(max(allocation_tangency, 0), sum(weights))
#             allocation_risk_free = max(sum(weights) - allocation_tangency, 0)

#             st.subheader("Portfolio Performance with Risk-Free Asset:")
#             st.write(f"Expected Annual Return: {portfolio_return:.2%}")
#             st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
#             st.write(f"Max tangency mean returns: {tangency_mean_returns.max()}")
#             st.write(f"Max tangency weights: {weights.max()}")
#             st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
#             st.write(
#                 f"Invest {allocation_tangency * 100:.2f}% in the tangency portfolio."
#             )
#             st.write(
#                 f"Invest {allocation_risk_free * 100:.2f}% in the risk-free asset."
#             )

#             # Show the allocation
#             allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
#             # Create a mapping of ISIN to Company name from static_data
#             isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))

#             # Replace ISIN in allocation_df with the corresponding company names
#             allocation_df["ISIN"] = allocation_df["ISIN"].map(isin_to_company)

#             # Optionally rename the column to reflect the new data
#             allocation_df.rename(columns={"ISIN": "Company"})

#             st.subheader("Tangency Portfolio Weights:")
#             st.write(allocation_df)

#         else:
#             # Calculate Sharpe Ratio
#             sharpe_ratio = portfolio_return / portfolio_volatility

#             st.subheader("Portfolio Performance without Risk-Free Asset:")
#             st.write(f"Expected Annual Return: {portfolio_return:.2%}")
#             st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
#             st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

#             # Show the allocation
#             allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
#             # Create a mapping of ISIN to Company name from static_data
#             isin_to_company = dict(zip(static_data["ISIN"], static_data["Company"]))

#             # Replace ISIN in allocation_df with the corresponding company names
#             allocation_df["ISIN"] = allocation_df["ISIN"].map(isin_to_company)

#             # Optionally rename the column to reflect the new data
#             allocation_df.rename(columns={"ISIN": "Company"})

#             st.write("Optimal Portfolio Allocation:")
#             st.write(allocation_df)
#         st.write(f"Sum of the weights: {np.sum(weights)}")
#     else:
#         st.write('Click "Run Optimization" to compute the optimized portfolio.')

#     # Run the efficient frontier
#     if st.session_state["optimization_run"]:
#         if st.button("Show Efficient Frontier"):
#             # Retrieve necessary variables from session state
#             weights = st.session_state["weights"]
#             mean_returns = st.session_state["mean_returns"]
#             cov_matrix = st.session_state["cov_matrix"]
#             optimized_returns = st.session_state["optimized_returns"]
#             optimized_volatility = st.session_state["optimized_volatility"]

#             st.write(f"Max mean returns for plot: {mean_returns.max()}")
#             num_assets = len(mean_returns)
#             weights_optimal = weights.values

#             plot_efficient_frontier(
#                 mean_returns,
#                 cov_matrix,
#                 risk_free_rate,
#                 include_risk_free_asset,
#                 weights_optimal,
#                 long_only,
#                 leverage_limit_value,
#                 min_weight_value,
#                 max_weight_value,
#                 num_portfolios=5000,
#             )
#         else:
#             st.write('Click "Show Efficient Frontier" to display the graph.')
#     else:
#         st.write("Run the optimization first to display the efficient frontier.")

# ---------------------------------------------
# FX Hedging Module for Streamlit
# ---------------------------------------------

def calculate_fx_exposure(portfolio_df, fx_rates, target_currency="USD"):
    """Calculates exposure to foreign currencies relative to target currency."""
    exposure = {}
    for currency in portfolio_df['Currency'].unique():
        if currency != target_currency:
            total_value_in_currency = portfolio_df[portfolio_df['Currency'] == currency]['Value'].sum()
            exposure[currency] = total_value_in_currency
    return exposure

def calculate_forward_rate(spot_rate, domestic_interest_rate, foreign_interest_rate, period_in_years):
    """Calculates the forward rate based on interest rate parity."""
    forward_rate = spot_rate * ((1 + domestic_interest_rate) / (1 + foreign_interest_rate)) ** period_in_years
    return forward_rate

def apply_fx_hedge(portfolio_df, fx_rates, hedge_currency, hedge_amount, hedge_duration, domestic_rate, foreign_rate):
    """Applies an FX hedge and calculates the effect on the portfolio's value."""
    spot_rate = fx_rates.get(hedge_currency, 1.0)
    forward_rate = calculate_forward_rate(spot_rate, domestic_rate, foreign_rate, hedge_duration)
    hedge_value_in_target = hedge_amount * (spot_rate - forward_rate)
    return hedge_value_in_target

# Streamlit App Section for FX Hedging
st.header("FX Hedging Options")

# Example portfolio and FX rate data (this should be dynamically loaded in real applications)
portfolio_df = pd.DataFrame({
    "Asset": ["Stock A", "Bond B"],
    "Currency": ["EUR", "GBP"],
    "Value": [10000, 5000]
})

fx_rates = {"EUR": 1.1, "GBP": 1.3}  # Example FX rates for EUR/USD and GBP/USD

# FX Exposure Calculation and Display
fx_exposure = calculate_fx_exposure(portfolio_df, fx_rates)
st.write("FX Exposure:", fx_exposure)

# Interactive inputs for hedging parameters
if fx_exposure:
    hedge_currency = st.selectbox("Select Currency to Hedge", options=fx_exposure.keys())
    hedge_amount = st.number_input("Hedge Amount (in foreign currency)", min_value=0.0, value=float(fx_exposure.get(hedge_currency, 0)))
    hedge_duration = st.slider("Hedge Duration (years)", min_value=0.0, max_value=5.0, step=0.1, value=1.0)
    domestic_rate = st.number_input("Domestic Interest Rate", min_value=0.0, max_value=1.0, step=0.01, value=0.02)
    foreign_rate = st.number_input("Foreign Interest Rate", min_value=0.0, max_value=1.0, step=0.01, value=0.01)

    # Calculate and display hedge effect
    hedge_effect = apply_fx_hedge(portfolio_df, fx_rates, hedge_currency, hedge_amount, hedge_duration, domestic_rate, foreign_rate)
    st.write("Hedge Effect on Portfolio Value:", hedge_effect)
else:
    st.write("No foreign currency exposure detected in the portfolio.")
