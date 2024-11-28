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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

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
        st.error(f"The file can't be read {bin_file}, info error: {e}")
        return ""


def set_background(image_path, target='page'):
    base64_img = get_base64_of_bin_file(image_path)
    if base64_img:
        if target == 'page':
            style = f'''
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
        elif target == 'sidebar':
            style = f'''
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background-image: url("data:image/jpeg;base64,{base64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            '''
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.warning(f"The background image at {image_path} could not be read.")

dir_path = os.path.dirname(os.path.realpath(__file__))

page_bg_img_path = os.path.join(dir_path, 'Static', 'BGP.jpg')
set_background(page_bg_img_path, target='page')

sidebar_bg_img_path = os.path.join(dir_path, 'Static', 'sidebar.jpg')

set_background(sidebar_bg_img_path, target='sidebar')


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
    leverage_limit_constraint_type,
    net_exposure,
    net_exposure_value,
    net_exposure_constraint_type,
    min_weight_value,
    max_weight_value,
    frontier_returns=None,
    frontier_volatility=None,
    frontier_weights=None,
):
    #reset the style as defaut, the backgroud is white
    plt.style.use('default')
    plt.figure(figsize=(10, 7))

    # be sure the efficient frontier data exists
    if (
        frontier_returns is None
        or frontier_volatility is None
        or frontier_weights is None
    ):
        tangency_return = np.sum(mean_returns * weights_optimal)

        # culculate the efficient frontier
        frontier_volatility, frontier_returns, frontier_weights = (
            calculate_efficient_frontier_qp(
                mean_returns,
                cov_matrix,
                long_only,
                include_risk_free_asset,
                risk_free_rate,
                leverage_limit,
                leverage_limit_value,
                leverage_limit_constraint_type,
                net_exposure,
                net_exposure_value,
                net_exposure_constraint_type,
                min_weight_value,
                max_weight_value,
                tangency_return,
            )
        )
    else:
        st.info("Using precomputed efficient frontier data.")

    # plot the efficient frontier with dash line
    plt.plot(
        frontier_volatility,
        frontier_returns,
        linestyle='-',
        color='black',
        linewidth=2,
        label='Efficient Frontier'
    )

    # plot individual assets which are colored by Sharp Ratio
    assets = mean_returns.index.tolist()
    asset_returns = mean_returns.values
    asset_volatility = np.sqrt(np.diag(cov_matrix.values))

    # culculate the SR of each asset
    asset_sharpe_ratios = (asset_returns - risk_free_rate) / asset_volatility

    # desgine the color from red to white to blue
    cmap = mcolors.LinearSegmentedColormap.from_list('sharpe_cmap', ['red', 'white', 'blue'])

    # normalize the max and min by SR
    norm = plt.Normalize(vmin=min(asset_sharpe_ratios), vmax=max(asset_sharpe_ratios))

    # color by SR
    colors = cmap(norm(asset_sharpe_ratios))

    # plot asset points
    scatter = plt.scatter(
        asset_volatility,
        asset_returns,
        marker='o',
        edgecolors='darkblue',
        facecolors=colors,
        s=50,
        alpha=0.8,
        linewidths=0.5,
        label='Individual Assets'
    )

    # add color bar to present SR
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio')
    
    # set the color of color bar as the same color range as points'
    cbar.mappable.set_cmap(cmap)
    cbar.mappable.set_norm(norm)

    # plot CML by black dash line
    if include_risk_free_asset:
        tangency_weights = weights_optimal
        tangency_return = np.sum(mean_returns * tangency_weights)
        tangency_volatility = np.sqrt(
            np.dot(tangency_weights.T, np.dot(cov_matrix.values, tangency_weights))
        )

        cml_x = [0, tangency_volatility]
        cml_y = [risk_free_rate, tangency_return]
        plt.plot(
            cml_x,
            cml_y,
            linestyle='--',
            color='darkred',
            linewidth=1.5,
            label='Capital Market Line'
        )

        # plot tangency portfolio by red star
        plt.scatter(
            tangency_volatility,
            tangency_return,
            marker='*',
            color='red',
            s=200,
            label='Tangency Portfolio'
        )
    else:
        # plot optimized portfolio
        portfolio_return = np.sum(mean_returns * weights_optimal)
        portfolio_volatility = np.sqrt(
            np.dot(weights_optimal.T, np.dot(cov_matrix.values, weights_optimal))
        )
        plt.scatter(
            portfolio_volatility,
            portfolio_return,
            marker='*',
            color='red',
            s=200,
            label='Optimal Portfolio'
        )

    # sed title and labels
    plt.title(
        'Efficient Frontier',
        fontsize=10, 
        loc='center',  
        fontweight='bold'  
    )      
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Expected Returns')

    # add grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # modify the appereance
    plt.legend(frameon=False, fontsize=10)

    # show the graph
    st.pyplot(plt)
    plt.close()



def plot_asset_allocation_bar_chart(weights, asset_names):
    
    df_weights = pd.DataFrame({"ISIN": asset_names, "Weight": weights})

    
    df_weights = df_weights.merge(
        static_data[["ISIN", "Company"]], on="ISIN", how="left"
    )

    df_weights["AbsWeight"] = df_weights["Weight"].abs()

    df_weights = df_weights.sort_values("AbsWeight", ascending=False)

    top_n = 20
    df_top = df_weights.head(top_n).copy()

    df_rest = df_weights.iloc[top_n:]
    other_weight = df_rest["Weight"].sum()
    other_abs_weight = df_rest["AbsWeight"].sum()
    if not df_rest.empty:
        df_other = pd.DataFrame(
            {
                "Company": ["Other"],
                "Weight": [other_weight],
                "AbsWeight": [other_abs_weight],
            }
        )
        df_top = pd.concat([df_top, df_other], ignore_index=True)

    y_pos = np.arange(len(df_top))
    bar_height = 0.6  

    fig, ax = plt.subplots(figsize=(12, 10))

    base_colors = ["green" if x >= 0 else "red" for x in df_top["Weight"]]

    bars = ax.barh(
        y_pos,
        df_top["Weight"],
        color='white',  
        edgecolor='none',  
        height=bar_height,
        left=0  
    )

    for bar, color in zip(bars, base_colors):
        width = bar.get_width()
        y = bar.get_y()
        height = bar.get_height()


        gradient = np.linspace(0, 1, 256).reshape(1, -1)

        if color == 'green':
            cmap = LinearSegmentedColormap.from_list('gradient_green', ['white', 'green'])
        else:
            cmap = LinearSegmentedColormap.from_list('gradient_red', ['white', 'red'])

        ax.imshow(
            gradient,
            extent=[0, width, y, y + height],
            aspect='auto',
            cmap=cmap,
            alpha=0.9,  
            zorder=1  
        )

    ax.axvline(x=0, color='black', linewidth=1.35, linestyle='-')

    padding = 0.5  
    ax.set_ylim(-padding, len(df_top) - 1 + padding)  

    ax.set_xlim(-0.4, 0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["Company"], fontsize=10)

    ax.set_xlabel("Weight")
    ax.set_ylabel("Company")
    ax.set_title(
            "Asset Allocation by Weight (Top Absolute Weights)",
            fontsize=10,  
            loc='center',  
            fontweight='bold'  
        )

    ax.invert_yaxis()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.7, axis='x')
    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.7, axis='y')

    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


def plot_weights_by_country(weights, asset_names):
    # Convert weights to percentages
    weights_percent = weights * 100
    # Create DataFrame
    df_weights = pd.DataFrame({"ISIN": asset_names, "Weight (%)": weights_percent})
    # Merge with static_data to get countries
    df_weights = df_weights.merge(
        static_data[["ISIN", "Country"]], on="ISIN", how="left"
    )
    # Group by country
    df_country = df_weights.groupby("Country")["Weight (%)"].sum().reset_index()
    # Sort countries by weight
    df_country = df_country.sort_values("Weight (%)", ascending=False)
    # Select top countries
    top_n = 10
    df_top = df_country.head(top_n).copy()
    # Sum the rest into 'Other'
    other_weight = df_country["Weight (%)"].iloc[top_n:].sum()
    if other_weight > 0:
        df_other = pd.DataFrame({"Country": ["Other"], "Weight (%)": [other_weight]})
        df_top = pd.concat([df_top, df_other], ignore_index=True)
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        df_top["Weight (%)"],
        labels=df_top["Country"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Portfolio Allocation by Country")
    plt.axis("equal")
    st.pyplot(plt)
    plt.close()


def plot_weights_by_carbon_emissions(weights, asset_names):
    # Convert weights to percentages
    weights_percent = weights * 100
    # Create DataFrame
    df_weights = pd.DataFrame({"ISIN": asset_names, "Weight (%)": weights_percent})
    # Merge with static_data to get carbon emissions
    df_weights = df_weights.merge(
        static_data[["ISIN", "TotalCarbonEmissions"]], on="ISIN", how="left"
    )
    # Handle missing values
    df_weights["TotalCarbonEmissions"] = df_weights["TotalCarbonEmissions"].fillna(0)
    # Categorize emissions into bins
    bins = [0, 1000, 5000, 10000, 50000, np.inf]
    labels = ["0-1k", "1k-5k", "5k-10k", "10k-50k", ">50k"]
    df_weights["EmissionCategory"] = pd.cut(
        df_weights["TotalCarbonEmissions"], bins=bins, labels=labels
    )
    # Group by emission category
    df_emission = (
        df_weights.groupby("EmissionCategory")["Weight (%)"].sum().reset_index()
    )
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        df_emission["Weight (%)"],
        labels=df_emission["EmissionCategory"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Portfolio Allocation by Carbon Emissions")
    plt.axis("equal")
    st.pyplot(plt)
    plt.close()


def plot_weights_by_carbon_intensity(weights, asset_names):
    # Convert weights to percentages
    weights_percent = weights * 100
    # Create DataFrame
    df_weights = pd.DataFrame({"ISIN": asset_names, "Weight (%)": weights_percent})
    # Merge with static_data to get carbon intensity
    df_weights = df_weights.merge(
        static_data[["ISIN", "CarbonIntensity"]], on="ISIN", how="left"
    )
    # Handle missing values
    df_weights["CarbonIntensity"] = df_weights["CarbonIntensity"].fillna(0)
    # Categorize intensity into bins
    bins = [0, 100, 500, 1000, 5000, np.inf]
    labels = ["0-100", "100-500", "500-1k", "1k-5k", ">5k"]
    df_weights["IntensityCategory"] = pd.cut(
        df_weights["CarbonIntensity"], bins=bins, labels=labels
    )
    # Group by intensity category
    df_intensity = (
        df_weights.groupby("IntensityCategory")["Weight (%)"].sum().reset_index()
    )
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        df_intensity["Weight (%)"],
        labels=df_intensity["IntensityCategory"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Portfolio Allocation by Carbon Intensity")
    plt.axis("equal")
    st.pyplot(plt)
    plt.close()


# def plot_asset_allocation_bar_chart(weights, asset_names):
#     # Convert weights to percentages
#     weights_percent = weights * 100
#     # Create a DataFrame for plotting
#     df_weights = pd.DataFrame({"ISIN": asset_names, "Weight (%)": weights_percent})

#     # Map ISINs to company names
#     df_weights = df_weights.merge(
#         static_data[["ISIN", "Company"]], on="ISIN", how="left"
#     )

#     # Sort by weight descending
#     df_weights = df_weights.sort_values("Weight (%)", ascending=False)

#     # Select top 20 weights
#     df_top = df_weights.head(20).copy()
#     # Sum the rest into "Other"
#     other_weight = df_weights["Weight (%)"].iloc[20:].sum()
#     if other_weight > 0:
#         df_other = pd.DataFrame({"Company": ["Other"], "Weight (%)": [other_weight]})
#         df_top = pd.concat([df_top, df_other], ignore_index=True)

#     # Plot bar chart
#     plt.figure(figsize=(12, 6))
#     plt.bar(df_top["Company"], df_top["Weight (%)"])
#     plt.xlabel("Company")
#     plt.ylabel("Weight (%)")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     st.pyplot(plt)
#     plt.close()

def plot_asset_risk_contribution(weights, cov_matrix):
    weights = np.array(weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    marginal_contrib = np.dot(cov_matrix.values, weights)
    risk_contrib = weights * marginal_contrib
    risk_contrib_percent = risk_contrib / portfolio_variance * 100

    asset_names = cov_matrix.columns.tolist()
    df_risk_contrib = pd.DataFrame(
        {"ISIN": asset_names, "Risk Contribution (%)": risk_contrib_percent}
    )

    df_risk_contrib = df_risk_contrib.merge(
        static_data[["ISIN", "Company"]], on="ISIN", how="left"
    )

    df_risk_contrib["AbsRiskContribution"] = df_risk_contrib["Risk Contribution (%)"].abs()

    df_risk_contrib = df_risk_contrib.sort_values("AbsRiskContribution", ascending=False)

    top_n = 20
    df_top = df_risk_contrib.head(top_n).copy()

    other_contrib = df_risk_contrib["Risk Contribution (%)"].iloc[top_n:].sum()
    other_abs_contrib = df_risk_contrib["AbsRiskContribution"].iloc[top_n:].sum()
    if not df_risk_contrib.iloc[top_n:].empty:
        df_other = pd.DataFrame(
            {
                "Company": ["Other"],
                "Risk Contribution (%)": [other_contrib],
                "AbsRiskContribution": [other_abs_contrib],
            }
        )
        df_top = pd.concat([df_top, df_other], ignore_index=True)

    y_pos = np.arange(len(df_top))
    bar_height = 0.6  

    fig, ax = plt.subplots(figsize=(12, 10))

    base_colors = ["blue" if x >= 0 else "red" for x in df_top["Risk Contribution (%)"]]

    bars = ax.barh(
        y_pos,
        df_top["Risk Contribution (%)"],
        color='white',  
        edgecolor='none',  
        height=bar_height,
        left=0  
    )

    for bar, color in zip(bars, base_colors):
        width = bar.get_width()
        y = bar.get_y()
        height = bar.get_height()

        gradient = np.linspace(0, 1, 256).reshape(1, -1)

        if color == 'blue':
            cmap = LinearSegmentedColormap.from_list('gradient_blue', ['white', 'blue'])
        else:
            cmap = LinearSegmentedColormap.from_list('gradient_red', ['white', 'red'])

        ax.imshow(
            gradient,
            extent=[0, width, y, y + height],
            aspect='auto',
            cmap=cmap,
            alpha=0.9, 
            zorder=1  
        )

    ax.axvline(x=0, color='black', linewidth=1.35, linestyle='-')

    padding = 0.5  
    ax.set_ylim(-padding, len(df_top) - 1 + padding)  

    min_weight = df_top["Risk Contribution (%)"].min()
    max_weight = df_top["Risk Contribution (%)"].max()
    margin = (max_weight - min_weight) * 0.1  
    ax.set_xlim(min_weight - margin, max_weight + margin)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["Company"], fontsize=10)

    ax.set_xlabel(
            "Risk Contribution (%)",
            loc='center'     
            )
    ax.set_ylabel(
            "Company",
            loc='center'
            )
    ax.set_title(
            "Asset Contribution to Portfolio Risk (Top Absolute Contributions)",
            fontsize=10, 
            loc='center',  
            fontweight='bold'  
        )

    ax.invert_yaxis()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.7, axis='x')

    
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


# -------------------------------
# Run the App
# -------------------------------

if __name__ == "__main__":
    main()
