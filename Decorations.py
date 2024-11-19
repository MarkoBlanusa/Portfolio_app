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


# New libraries #
import os
import base64

# Decorations for buttons # 
global_button_css = """
<style>
    div.stButton > button {
        position: relative;
        overflow: hidden;
        padding: 10px 20px;
        font-size: 16px;
        color: white;
        background: linear-gradient(90deg, rgba(128, 128, 128, 0.65), rgba(96, 96, 96, 0.65)); 
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    div.stButton > button:hover {
        transform: scale(1.2);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    div.stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3); 
        border-radius: 50%;
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
        transition: width 0.6s ease, height 0.6s ease, opacity 0.6s ease;
    }

    div.stButton > button:active::after {
        width: 300px;
        height: 300px;
        opacity: 0; 
    }

    div.stNumberInput > div > button {
        all: unset;
    }
</style>
"""

st.markdown(global_button_css, unsafe_allow_html=True)



# Add Background Picture #
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"The fiel can't be read {bin_file},info error:{e}")
        return ""

dir_path = os.path.dirname(os.path.realpath(__file__))

img_path = os.path.join(dir_path, 'Static', 'BGP.jpg')

base64_img = get_base64_of_bin_file(img_path)

if base64_img:
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
else:
    st.warning("The background picture can't be read, use the defaut picture.")