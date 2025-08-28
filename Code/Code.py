# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 12:43:44 2025

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from pytrends.request import TrendReq
import os
import time
import scipy.stats as stats

# Load dataset
df = pd.read_excel("Online Retail.xlsx")

# Clean the data
df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Add helper columns
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Sales'] = df['Quantity'] * df['UnitPrice']
df['Week'] = df['InvoiceDate'].dt.isocalendar().week

# Focus on top products for simplicity
top_products = df.groupby('Description')['Sales'].sum().nlargest(5).index
df = df[df['Description'].isin(top_products)]

# Aggregate to weekly product-level data
weekly_data = df.groupby(['Week', 'Description']).agg({
    'Quantity': 'sum',
    'Sales': 'sum',
    'UnitPrice': 'mean'
}).reset_index()

# Calculate average price
weekly_data['AvgPrice'] = weekly_data['Sales'] / weekly_data['Quantity']

# --------------------------
# Generate synthetic trend data
# --------------------------

# Get the unique weeks from the retail dataset
weeks = sorted(weekly_data['Week'].unique())

# NOTE: Originally intended to combine Twitter sentiment with Google Trends.
# Due to API and time constraints, this uses only synthetic sentiment data
# to simulate interest volatility. Twitter sentiment was omitted.

# Simulate sentiment values (base + noise)
np.random.seed(42)  # For reproducibility
base_trend = np.linspace(0.4, 0.7, len(weeks))  # Slowly increasing interest
noise = np.random.normal(0, 0.05, len(weeks))   # Random volatility
synthetic_sentiment = np.clip(base_trend + noise, 0, 1)

# Create synthetic trend DataFrame
trend_df = pd.DataFrame({
    'Week': weeks,
    'sentiment_score': synthetic_sentiment
})

# Normalize to [0, 1] to form the final TrendScore
scaler = MinMaxScaler()
trend_df['TrendScore'] = scaler.fit_transform(trend_df[['sentiment_score']])

print("Retail weeks:", weekly_data['Week'].nunique())
print("Trend weeks: ", trend_df['Week'].nunique())
 
# Merge with retail data
retail_trend_data = pd.merge(weekly_data, trend_df[['Week', 'TrendScore']], on='Week', how='left')

# Sanity check
print(retail_trend_data.head())


# Step 1: Create RefPrice if not already there
retail_trend_data['RefPrice'] = retail_trend_data.groupby('Week')['AvgPrice'].shift(1)

# Step 2: Drop missing values
retail_trend_data = retail_trend_data.dropna(subset=['RefPrice'])

# Step 3: One-hot encode products
if 'Description' in retail_trend_data.columns:
    if retail_trend_data['Description'].dtype == object:
        df_encoded = pd.get_dummies(retail_trend_data, columns=['Description'], drop_first=True)
    else:
        df_encoded = retail_trend_data.copy()
else:
    df_encoded = retail_trend_data.copy()


# Step 4: Prepare features
X = df_encoded[['AvgPrice', 'RefPrice', 'TrendScore'] + [col for col in df_encoded.columns if col.startswith('Description_')]]

y = df_encoded['Quantity']

# Step 5: Ensure all data is numeric and clean
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Step 6: Drop NaNs from both X and y
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]

X = X.astype(float)
y = y.astype(float)

# Step 7: Add intercept
X = sm.add_constant(X)

# Step 8: Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Make sure X includes the constant already
X_with_const = sm.add_constant(X, has_constant='add')

# Create a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data['feature'] = X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) 
                   for i in range(X_with_const.shape[1])]

print(vif_data)

# Step 9: Create PriceChange
df_encoded['PriceChange'] = df_encoded['AvgPrice'] - df_encoded['RefPrice']

# Step 10: Drop problematic dummies (if they exist)
cols_to_drop = [
    'Description_PARTY BUNTING',
    'Description_REGENCY CAKESTAND 3 TIER',
    'Description_WHITE HANGING HEART T-LIGHT HOLDER'
]
df_encoded = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns])

# Step 11: Rebuild X with PriceChange instead of AvgPrice & RefPrice
X = df_encoded[['PriceChange', 'TrendScore'] +
               [col for col in df_encoded.columns if col.startswith('Description_')]]

# Step 12: Ensure numeric and clean again
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df_encoded['Quantity'], errors='coerce')
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask].astype(float)
y = y[mask].astype(float)

# Step 13: Add intercept and fit model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Step 14: Check VIFs again
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

#Simulation
# Load trained model
with open("demand_model.pkl", "rb") as f:
    model = pickle.load(f)

#For now 3 firms only 
firms = {
    'Firm_A': {'strategy': 'static', 'base_price': 10},
    'Firm_B': {'strategy': 'markdown', 'base_price': 10, 'markdown_week': 5, 'markdown_price': 8},
    'Firm_C': {'strategy': 'promotion', 'base_price': 10, 'promo_weeks': [3, 6], 'promo_price': 7}
}


# Extract model features (including 'const')
model_features = model.model.exog_names

# Simulate trend for 10 weeks
np.random.seed(42)
weeks = list(range(1, 11))
base_trend = np.linspace(0.4, 0.7, len(weeks)) + np.random.normal(0, 0.05, len(weeks))
trend_scores = MinMaxScaler().fit_transform(base_trend.reshape(-1, 1)).flatten()

# Init results and price memory
results = []
last_prices = {}

for week_idx, week in enumerate(weeks):
    trend_score = trend_scores[week_idx]

    for firm_name, firm in firms.items():
        base_price = firm['base_price']

        # Determine price based on strategy
        if firm['strategy'] == 'static':
            price = base_price
        elif firm['strategy'] == 'markdown':
            price = firm['markdown_price'] if week >= firm['markdown_week'] else base_price
        elif firm['strategy'] == 'promotion':
            price = firm['promo_price'] if week in firm['promo_weeks'] else base_price
        else:
            price = base_price

        # Use last week's price or fallback to base
        ref_price = base_price if week == 1 else last_prices.get(firm_name, base_price)
        price_change = price - ref_price

        # Build data dict with all required features from model
        data = {col: 0.0 for col in model_features}  # initialize all to 0
        data['const'] = 1.0
        data['PriceChange'] = price_change
        data['TrendScore'] = trend_score

        # Create DataFrame and predict
        X_sim = pd.DataFrame([data])[model_features]
        quantity = model.predict(X_sim)[0]
        revenue = price * quantity

        results.append({
            'Week': week,
            'Firm': firm_name,
            'Price': price,
            'RefPrice': ref_price,
            'TrendScore': trend_score,
            'PredictedQuantity': quantity,
            'Revenue': revenue
        })

        last_prices[firm_name] = price  # Save price for next week's reference

results_df = pd.DataFrame(results)
print(results_df.head())


for firm in firms:
    plt.plot(results_df[results_df['Firm'] == firm]['Week'],
             results_df[results_df['Firm'] == firm]['Revenue'],
             label=firm)

plt.xlabel("Week")
plt.ylabel("Revenue")
plt.title("Firm Revenues Over Time")
plt.legend()
plt.grid(True)
plt.show()

results_df.groupby("Firm")["Revenue"].sum()

results_df.groupby("Firm")["PredictedQuantity"].mean()


##Shared Demand Pool
def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

results = []
last_prices = {}
unit_cost = 5  # assumed cost

for week_idx, week in enumerate(weeks):
    trend_score = trend_scores[week_idx]
    firm_prices = {}
    firm_utils = {}

    # First pass: compute each firm's price and utility
    for firm_name, firm in firms.items():
        base_price = firm['base_price']
        if firm['strategy'] == 'static':
            price = base_price
        elif firm['strategy'] == 'markdown':
            price = firm['markdown_price'] if week >= firm['markdown_week'] else base_price
        elif firm['strategy'] == 'promotion':
            price = firm['promo_price'] if week in firm['promo_weeks'] else base_price
        else:
            price = base_price

        ref_price = base_price if week == 1 else last_prices.get(firm_name, base_price)
        price_change = price - ref_price

        # Utility function: lower price and higher trend = better
        alpha = 1.0
        beta = 1.0
        utility = -alpha * price + beta * trend_score

        firm_prices[firm_name] = {
            'price': price,
            'ref_price': ref_price,
            'price_change': price_change
        }
        firm_utils[firm_name] = utility

    # Convert utilities to market shares using softmax
    utilities = np.array(list(firm_utils.values()))
    market_shares = softmax(utilities)
    total_demand = 1500 + 500 * trend_score  # total market size for this week

    for i, (firm_name, firm) in enumerate(firms.items()):
        share = market_shares[i]
        quantity = total_demand * share
        price = firm_prices[firm_name]['price']
        revenue = price * quantity
        profit = (price - unit_cost) * quantity

        results.append({
            'Week': week,
            'Firm': firm_name,
            'Price': price,
            'RefPrice': firm_prices[firm_name]['ref_price'],
            'TrendScore': trend_score,
            'MarketShare': share,
            'PredictedQuantity': quantity,
            'Revenue': revenue,
            'Profit': profit
        })

        last_prices[firm_name] = price

results_df = pd.DataFrame(results)

# Plot revenue over time
for firm in firms:
    firm_data = results_df[results_df['Firm'] == firm]
    plt.plot(firm_data['Week'], firm_data['Revenue'], label=firm)

plt.title("Weekly Revenue (Shared Demand)")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()

# View summary
print(results_df.groupby("Firm")[["Revenue", "Profit", "PredictedQuantity"]].sum())


## Simulate trend shocks (viral spike, decay, noise)
np.random.seed(42)
weeks = list(range(1, 11))

# Base trend curve
trend = np.linspace(0.4, 0.7, len(weeks))

# Add viral spike at week 4, decay after week 7
shock = np.zeros(len(weeks))
shock[3] = 0.5  # viral spike (week 4)
shock[7:] = -0.3  # fading interest (week 8+)

# Add noise
noise = np.random.normal(0, 0.05, len(weeks))

# Combine everything
synthetic_trend = trend + shock + noise
synthetic_trend = np.clip(synthetic_trend, 0, 1)  # cap between 0 and 1

# Normalize to 0–1 range
trend_scores = MinMaxScaler().fit_transform(synthetic_trend.reshape(-1, 1)).flatten()

# Optional: plot to visualize
plt.plot(weeks, trend_scores, marker='o', linestyle='--', color='purple')
plt.title("Simulated Trend Score (with Shock)")
plt.xlabel("Week")
plt.ylabel("TrendScore")
plt.grid(True)
plt.show()

##Bayesian Trend Belief Update 
def bayesian_trend_update(observed, prior_mean, prior_var, obs_var):
    # Compute posterior variance
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    
    # Compute posterior mean
    posterior_mean = (obs_var * prior_mean + prior_var * observed) / (prior_var + obs_var)
    
    return posterior_mean, posterior_var

# Set parameters
prior_mean = 0.5  # Initial belief
prior_var = 0.05  # Initial uncertainty
obs_var = 0.02    # Noise in observation

belief_means = []
belief_vars = []

for week_idx, obs in enumerate(trend_scores):
    posterior_mean, posterior_var = bayesian_trend_update(
        observed=obs,
        prior_mean=prior_mean,
        prior_var=prior_var,
        obs_var=obs_var
    )

    belief_means.append(posterior_mean)
    belief_vars.append(posterior_var)

    # Posterior becomes prior for next week
    prior_mean = posterior_mean
    prior_var = posterior_var

plt.figure(figsize=(10,5))
plt.plot(weeks, trend_scores, label="Observed TrendScore", linestyle='--', marker='o')
plt.plot(weeks, belief_means, label="Bayesian Belief", linestyle='-', marker='s')
plt.fill_between(weeks,
                 np.array(belief_means) - 1.96 * np.sqrt(belief_vars),
                 np.array(belief_means) + 1.96 * np.sqrt(belief_vars),
                 color='orange', alpha=0.3, label="95% Confidence")
plt.title("Bayesian Trend Belief Update Over Time")
plt.xlabel("Week")
plt.ylabel("TrendScore")
plt.legend()
plt.grid(True)
plt.show()

#TrendScore → Bayesian Belief in Shared Demand Simulation
# --- Step 1: Define Bayesian update function ---
def bayesian_trend_update(observed, prior_mean, prior_var, obs_var):
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    posterior_mean = (obs_var * prior_mean + prior_var * observed) / (prior_var + obs_var)
    return posterior_mean, posterior_var

# --- Step 2: Simulate raw trend signal with noise, spikes, and decay ---
np.random.seed(42)
weeks = list(range(1, 11))
base_trend = np.linspace(0.4, 0.7, len(weeks))
shock = np.zeros(len(weeks))
shock[3] = 0.5   # Viral spike (week 4)
shock[7:] = -0.3  # Trend decay (week 8+)
noise = np.random.normal(0, 0.05, len(weeks))
observed_trend = base_trend + shock + noise
observed_trend = np.clip(observed_trend, 0, 1)

# Normalize the observed trend to match earlier scales
trend_scores = MinMaxScaler().fit_transform(observed_trend.reshape(-1, 1)).flatten()

# --- Step 3: Apply Bayesian filtering to get smoothed trend belief ---
prior_mean = 0.5
prior_var = 0.05
obs_var = 0.02
belief_means = []
belief_vars = []

for obs in trend_scores:
    posterior_mean, posterior_var = bayesian_trend_update(obs, prior_mean, prior_var, obs_var)
    belief_means.append(posterior_mean)
    belief_vars.append(posterior_var)
    prior_mean, prior_var = posterior_mean, posterior_var

# --- Step 4: Shared Demand Simulation using Bayesian Belief ---
def softmax(x):
    e_x = np.exp(x - np.max(x))  # numerical stability
    return e_x / e_x.sum()

firms = {
    'Firm_A': {'strategy': 'static', 'base_price': 10},
    'Firm_B': {'strategy': 'markdown', 'base_price': 10, 'markdown_week': 5, 'markdown_price': 8},
    'Firm_C': {'strategy': 'promotion', 'base_price': 10, 'promo_weeks': [3, 6], 'promo_price': 7}
}

results = []
last_prices = {}
unit_cost = 5  # constant for all firms

for week_idx, week in enumerate(weeks):
    # Use Bayesian belief instead of raw TrendScore
    trend_score = belief_means[week_idx]

    firm_prices = {}
    firm_utils = {}

    # --- Determine pricing and utility for each firm ---
    for firm_name, firm in firms.items():
        base_price = firm['base_price']
        if firm['strategy'] == 'static':
            price = base_price
        elif firm['strategy'] == 'markdown':
            price = firm['markdown_price'] if week >= firm['markdown_week'] else base_price
        elif firm['strategy'] == 'promotion':
            price = firm['promo_price'] if week in firm['promo_weeks'] else base_price
        else:
            price = base_price

        ref_price = base_price if week == 1 else last_prices.get(firm_name, base_price)
        price_change = price - ref_price

        # Define utility function
        alpha = 1.0
        beta = 1.0
        utility = -alpha * price + beta * trend_score

        firm_prices[firm_name] = {
            'price': price,
            'ref_price': ref_price,
            'price_change': price_change
        }
        firm_utils[firm_name] = utility

    # --- Convert utilities into market shares ---
    utilities = np.array(list(firm_utils.values()))
    market_shares = softmax(utilities)
    total_demand = 1500 + 500 * trend_score

    for i, (firm_name, firm) in enumerate(firms.items()):
        share = market_shares[i]
        quantity = total_demand * share
        price = firm_prices[firm_name]['price']
        revenue = price * quantity
        profit = (price - unit_cost) * quantity

        results.append({
            'Week': week,
            'Firm': firm_name,
            'Price': price,
            'RefPrice': firm_prices[firm_name]['ref_price'],
            'TrendScore_Belief': trend_score,
            'MarketShare': share,
            'PredictedQuantity': quantity,
            'Revenue': revenue,
            'Profit': profit
        })

        last_prices[firm_name] = price

# --- Step 5: Results and Visualization ---
results_df = pd.DataFrame(results)

# Revenue plot
for firm in firms:
    firm_data = results_df[results_df['Firm'] == firm]
    plt.plot(firm_data['Week'], firm_data['Revenue'], label=firm)

plt.title("Weekly Revenue with Bayesian Belief Trend")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()

# Summary performance
print(results_df.groupby("Firm")[["Revenue", "Profit", "PredictedQuantity"]].sum())

##Reinforcement Learning 
# --- Define environment parameters ---
price_levels = [7, 8, 9, 10]  # Actions
trend_bins = np.linspace(0, 1, 5)  # State bins for trend
episodes = 200  # Number of learning iterations
epsilon = 0.2  # Exploration rate
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor

# --- Define other firms (fixed strategies) ---
firms_fixed = {
    'Firm_B': {'strategy': 'markdown', 'base_price': 10, 'markdown_week': 5, 'markdown_price': 8},
    'Firm_C': {'strategy': 'promotion', 'base_price': 10, 'promo_weeks': [3, 6], 'promo_price': 7}
}

# --- Initialize Q-table ---
Q = defaultdict(lambda: np.zeros(len(price_levels)))

# --- Helper functions ---
def get_state(trend_score, last_price):
    trend_bin = np.digitize(trend_score, trend_bins) - 1
    price_idx = price_levels.index(last_price)
    return (trend_bin, price_idx)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- Simulated market loop ---
reward_history = []

for episode in range(episodes):
    last_price = 10  # Starting price for Firm A
    total_profit = 0
    prior_mean = 0.5
    prior_var = 0.05
    obs_var = 0.02

    for week in range(10):
        # Bayesian belief update
        observed = trend_scores[week]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, obs_var)
        prior_mean = trend_belief

        # RL state
        state = get_state(trend_belief, last_price)

        # Action: Choose a price
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(price_levels))
        else:
            action_idx = np.argmax(Q[state])

        firm_a_price = price_levels[action_idx]
        price_change = firm_a_price - last_price
        last_price = firm_a_price  # Update for next week

        # Set firm prices
        firm_prices = {
            'Firm_A': firm_a_price
        }

        for firm, data in firms_fixed.items():
            base_price = data['base_price']
            if data['strategy'] == 'markdown':
                firm_prices[firm] = data['markdown_price'] if week >= data['markdown_week'] else base_price
            elif data['strategy'] == 'promotion':
                firm_prices[firm] = data['promo_price'] if week in data['promo_weeks'] else base_price

        # Compute utilities
        firm_utils = {}
        for firm, price in firm_prices.items():
            firm_utils[firm] = -price + trend_belief  # Simplified utility

        market_shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        quantities = {firm: share * total_demand for firm, share in zip(firm_prices.keys(), market_shares)}

        profit = (firm_a_price - 5) * quantities['Firm_A']
        total_profit += profit

        # Observe next state (we use same trend belief next step for simplicity here)
        next_state = get_state(trend_belief, last_price)

        # Q-learning update
        Q[state][action_idx] += alpha * (profit + gamma * np.max(Q[next_state]) - Q[state][action_idx])

    reward_history.append(total_profit)

# --- Plot Learning Curve ---
plt.plot(reward_history)
plt.title("RL Agent (Firm A) Profit per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Profit (10 weeks)")
plt.grid(True)
plt.show()

# Evaluate learned RL policy
final_results = []
last_price = 10
prior_mean = 0.5
prior_var = 0.05

for week in range(10):
    # Bayesian belief
    observed = trend_scores[week]
    trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, obs_var)
    prior_mean = trend_belief

    # RL action (greedy)
    state = get_state(trend_belief, last_price)
    action_idx = np.argmax(Q[state])
    firm_a_price = price_levels[action_idx]
    price_change = firm_a_price - last_price
    last_price = firm_a_price

    firm_prices = {'Firm_A': firm_a_price}
    for firm, data in firms_fixed.items():
        base_price = data['base_price']
        if data['strategy'] == 'markdown':
            firm_prices[firm] = data['markdown_price'] if week >= data['markdown_week'] else base_price
        elif data['strategy'] == 'promotion':
            firm_prices[firm] = data['promo_price'] if week in data['promo_weeks'] else base_price

    # Utilities and shares
    firm_utils = {firm: -p + trend_belief for firm, p in firm_prices.items()}
    market_shares = softmax(np.array(list(firm_utils.values())))
    total_demand = 1500 + 500 * trend_belief
    quantities = {firm: share * total_demand for firm, share in zip(firm_prices.keys(), market_shares)}

    for firm in firm_prices:
        price = firm_prices[firm]
        quantity = quantities[firm]
        revenue = price * quantity
        profit = (price - 5) * quantity
        final_results.append({
            'Week': week + 1,
            'Firm': firm,
            'Price': price,
            'TrendScore_Belief': trend_belief,
            'PredictedQuantity': quantity,
            'Revenue': revenue,
            'Profit': profit
        })

# Create DataFrame and summarize
final_df = pd.DataFrame(final_results)
summary = final_df.groupby("Firm")[["Revenue", "Profit", "PredictedQuantity"]].sum()
print(summary)

# Smooth learning curve
smoothed = pd.Series(reward_history).rolling(10).mean()
plt.plot(reward_history, alpha=0.3, label='Raw')
plt.plot(smoothed, label='Smoothed', color='blue')
plt.title("RL Learning Curve (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Total Profit (10 weeks)")
plt.legend()
plt.grid(True)
plt.show()


#Expanded State Space and Action = Strategy
# --- Define pricing strategy mapping ---
def get_price_from_strategy(strategy, week):
    if strategy == 0:  # Static
        return 10
    elif strategy == 1:  # Markdown
        return 8 if week >= 5 else 10
    elif strategy == 2:  # Promotion
        return 7 if week in [3, 6] else 10

# --- Binning helpers ---
price_bins = [7, 8, 9, 10]
trend_bins = np.linspace(0, 1, 5)
demand_bins = [0, 2000, 4000, 6000, 8000]

# --- New state function ---
def get_state(trend_score, last_price, ref_price, past_demand):
    t_bin = np.digitize(trend_score, trend_bins) - 1
    lp_bin = np.digitize(last_price, price_bins) - 1
    rp_bin = np.digitize(ref_price, price_bins) - 1
    d_bin = np.digitize(past_demand, demand_bins) - 1
    return (t_bin, lp_bin, rp_bin, d_bin)

# --- RL parameters ---
strategies = [0, 1, 2]  # 0 = Static, 1 = Markdown, 2 = Promo
Q = defaultdict(lambda: np.zeros(len(strategies)))

epsilon = 0.2
alpha = 0.1
gamma = 0.9
episodes = 200

reward_history = []

# --- Training loop ---
for episode in range(episodes):
    prior_mean = 0.5
    prior_var = 0.05
    last_price = 10
    ref_price = 10
    past_demand = 3000  # neutral starting point
    total_profit = 0

    for week in range(1, 11):
        # Trend belief update
        observed = trend_scores[week - 1]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief

        # Get current state
        state = get_state(trend_belief, last_price, ref_price, past_demand)

        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = np.random.choice(strategies)
        else:
            action = np.argmax(Q[state])

        # Translate strategy to price
        firm_a_price = get_price_from_strategy(action, week)
        price_change = firm_a_price - ref_price

        # Fixed strategies for Firm B and C
        firm_prices = {
            'Firm_A': firm_a_price,
            'Firm_B': 8 if week >= 5 else 10,
            'Firm_C': 7 if week in [3, 6] else 10
        }

        firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
        market_shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        quantities = {f: s * total_demand for f, s in zip(firm_prices, market_shares)}

        # Compute reward
        profit = (firm_a_price - 5) * quantities['Firm_A']
        total_profit += profit

        # Next state
        next_state = get_state(trend_belief, firm_a_price, last_price, quantities['Firm_A'])

        # Q-learning update
        Q[state][action] += alpha * (profit + gamma * np.max(Q[next_state]) - Q[state][action])

        # Update memory
        ref_price = last_price
        last_price = firm_a_price
        past_demand = quantities['Firm_A']

    reward_history.append(total_profit)

# --- Evaluation ---
final_results = []
prior_mean = 0.5
prior_var = 0.05
last_price = 10
ref_price = 10
past_demand = 3000

for week in range(1, 11):
    observed = trend_scores[week - 1]
    trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
    prior_mean = trend_belief

    state = get_state(trend_belief, last_price, ref_price, past_demand)
    strategy = np.argmax(Q[state])
    firm_a_price = get_price_from_strategy(strategy, week)
    price_change = firm_a_price - ref_price

    firm_prices = {
        'Firm_A': firm_a_price,
        'Firm_B': 8 if week >= 5 else 10,
        'Firm_C': 7 if week in [3, 6] else 10
    }

    firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
    market_shares = softmax(np.array(list(firm_utils.values())))
    total_demand = 1500 + 500 * trend_belief
    quantities = {f: s * total_demand for f, s in zip(firm_prices, market_shares)}

    for f in firm_prices:
        p = firm_prices[f]
        q = quantities[f]
        final_results.append({
            'Week': week,
            'Firm': f,
            'Price': p,
            'Quantity': q,
            'Revenue': p * q,
            'Profit': (p - 5) * q
        })

    ref_price = last_price
    last_price = firm_a_price
    past_demand = quantities['Firm_A']

# --- Results ---
df_eval = pd.DataFrame(final_results)
print(df_eval.groupby("Firm")[["Revenue", "Profit", "Quantity"]].sum())


##Run more episodes + Decay epsilon
# --- Extended RL training with epsilon decay ---
episodes = 500
epsilon = 1.0            # Start with full exploration
epsilon_min = 0.05       # Minimum allowed exploration
decay_rate = 0.995       # How fast epsilon shrinks

reward_history = []

for episode in range(episodes):
    prior_mean = 0.5
    prior_var = 0.05
    last_price = 10
    ref_price = 10
    past_demand = 3000
    total_profit = 0

    for week in range(1, 11):
        observed = trend_scores[week - 1]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief

        state = get_state(trend_belief, last_price, ref_price, past_demand)

        # Epsilon-decay action
        if np.random.rand() < epsilon:
            action = np.random.choice(strategies)
        else:
            action = np.argmax(Q[state])

        firm_a_price = get_price_from_strategy(action, week)
        price_change = firm_a_price - ref_price

        firm_prices = {
            'Firm_A': firm_a_price,
            'Firm_B': 8 if week >= 5 else 10,
            'Firm_C': 7 if week in [3, 6] else 10
        }

        firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
        market_shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        quantities = {f: s * total_demand for f, s in zip(firm_prices, market_shares)}

        profit = (firm_a_price - 5) * quantities['Firm_A']
        total_profit += profit

        next_state = get_state(trend_belief, firm_a_price, last_price, quantities['Firm_A'])

        Q[state][action] += alpha * (profit + gamma * np.max(Q[next_state]) - Q[state][action])

        ref_price = last_price
        last_price = firm_a_price
        past_demand = quantities['Firm_A']

    reward_history.append(total_profit)
    epsilon = max(epsilon_min, epsilon * decay_rate)  # Decay epsilon

print("Training complete. Final epsilon:", epsilon)

smoothed = pd.Series(reward_history).rolling(10).mean()
plt.plot(reward_history, alpha=0.3, label='Raw')
plt.plot(smoothed, label='Smoothed', color='blue')
plt.title("Extended RL Learning Curve with Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Total Profit (10 weeks)")
plt.legend()
plt.grid(True)
plt.show()

# KPI Summary
summary = df_eval.groupby("Firm").agg({
    "Revenue": "sum",
    "Profit": "sum",
    "Quantity": "sum",
    "Price": "mean"
}).rename(columns={"Quantity": "TotalQuantity", "Price": "AvgPrice"})

# Calculate ROI
summary["ROI"] = summary["Profit"] / summary["Revenue"]

# Calculate market share
total_quantity = summary["TotalQuantity"].sum()
summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / total_quantity

print("\n Key Performance Indicators (KPIs):")
print(summary.round(2))

#KPI summary for different RL versions 
#| Version Label          | Description                                    |
#| ---------------------- | ---------------------------------------------- |
#| RL_Basic               | Action = price; state = (trend, last price)    |
#| RL_Strategy            | Action = pricing strategy; expanded state      |
#| RL_Strategy+Decay      | Strategy action + epsilon decay + 500 episodes |
#| Rule_Based_B           | Firm B: markdown                               |
#| Rule_Based_C           | Firm C: promo                                  |


# === Utility Functions ===
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def bayesian_trend_update(observed, prior_mean, prior_var, obs_var):
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    posterior_mean = (obs_var * prior_mean + prior_var * observed) / (prior_var + obs_var)
    return posterior_mean, posterior_var

# === RL Basic Version Training ===
price_levels = [7, 8, 9, 10]
trend_bins = np.linspace(0, 1, 5)
Q_basic = defaultdict(lambda: np.zeros(len(price_levels)))

def get_state_basic(trend_score, last_price):
    t_bin = np.digitize(trend_score, trend_bins) - 1
    p_bin = price_levels.index(last_price)
    return (t_bin, p_bin)

for episode in range(200):
    prior_mean = 0.5
    prior_var = 0.05
    last_price = 10
    for week in range(10):
        observed = trend_scores[week]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief
        state = get_state_basic(trend_belief, last_price)
        epsilon = 0.2
        action_idx = np.random.choice(len(price_levels)) if np.random.rand() < epsilon else np.argmax(Q_basic[state])
        firm_a_price = price_levels[action_idx]
        last_price = firm_a_price

        firm_prices = {'Firm_A': firm_a_price, 'Firm_B': 8 if week >= 5 else 10, 'Firm_C': 7 if week in [3, 6] else 10}
        firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
        shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        quantities = {f: s * total_demand for f, s in zip(firm_prices, shares)}
        reward = (firm_a_price - 5) * quantities['Firm_A']
        next_state = get_state_basic(trend_belief, firm_a_price)
        Q_basic[state][action_idx] += 0.1 * (reward + 0.9 * np.max(Q_basic[next_state]) - Q_basic[state][action_idx])

# === RL Basic Evaluation ===
results_basic = []
prior_mean = 0.5
prior_var = 0.05
last_price = 10

for week in range(10):
    observed = trend_scores[week]
    trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
    prior_mean = trend_belief
    state = get_state_basic(trend_belief, last_price)
    action_idx = np.argmax(Q_basic[state])
    firm_a_price = price_levels[action_idx]
    last_price = firm_a_price

    firm_prices = {'Firm_A': firm_a_price, 'Firm_B': 8 if week >= 5 else 10, 'Firm_C': 7 if week in [3, 6] else 10}
    firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
    shares = softmax(np.array(list(firm_utils.values())))
    total_demand = 1500 + 500 * trend_belief
    quantities = {f: s * total_demand for f, s in zip(firm_prices, shares)}

    for f in firm_prices:
        p, q = firm_prices[f], quantities[f]
        results_basic.append({
            'Week': week + 1, 'Firm': f, 'Price': p,
            'Quantity': q, 'Revenue': p * q, 'Profit': (p - 5) * q
        })

df_eval_basic = pd.DataFrame(results_basic)

# === RL Strategy Evaluation Function ===
def get_price_from_strategy(strategy, week):
    if strategy == 0: return 10
    elif strategy == 1: return 8 if week >= 5 else 10
    elif strategy == 2: return 7 if week in [3, 6] else 10

def get_state(trend_score, last_price, ref_price, past_demand):
    price_bins = [7, 8, 9, 10]
    trend_bins = np.linspace(0, 1, 5)
    demand_bins = [0, 2000, 4000, 6000, 8000]
    t_bin = np.digitize(trend_score, trend_bins) - 1
    lp_bin = np.digitize(last_price, price_bins) - 1
    rp_bin = np.digitize(ref_price, price_bins) - 1
    d_bin = np.digitize(past_demand, demand_bins) - 1
    return (t_bin, lp_bin, rp_bin, d_bin)

def evaluate_rl_policy(Q, strategies, get_state, get_price_from_strategy):
    results = []
    prior_mean = 0.5
    prior_var = 0.05
    last_price = 10
    ref_price = 10
    past_demand = 3000

    for week in range(1, 11):
        observed = trend_scores[week - 1]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief
        state = get_state(trend_belief, last_price, ref_price, past_demand)
        strategy = np.argmax(Q[state])
        firm_a_price = get_price_from_strategy(strategy, week)
        ref_price = last_price
        last_price = firm_a_price

        firm_prices = {'Firm_A': firm_a_price, 'Firm_B': 8 if week >= 5 else 10, 'Firm_C': 7 if week in [3, 6] else 10}
        firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
        shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        quantities = {f: s * total_demand for f, s in zip(firm_prices, shares)}

        past_demand = quantities['Firm_A']
        for f in firm_prices:
            p, q = firm_prices[f], quantities[f]
            results.append({
                'Week': week, 'Firm': f, 'Price': p,
                'Quantity': q, 'Revenue': p * q, 'Profit': (p - 5) * q
            })

    return pd.DataFrame(results)

# === Evaluate Strategy-Based and Final RL (Assuming Q is final strategy+decay Q-table) ===
df_eval_strategy = evaluate_rl_policy(Q, strategies=[0, 1, 2],
                                      get_state=get_state,
                                      get_price_from_strategy=get_price_from_strategy)

df_eval_final = df_eval.copy()  # Assuming df_eval already created after final RL execution

# === KPI Calculation Function ===
def compute_kpis(df):
    summary = df.groupby("Firm").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Price": "mean"
    }).rename(columns={"Quantity": "TotalQuantity", "Price": "AvgPrice"})
    summary["ROI"] = summary["Profit"] / summary["Revenue"]
    summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / summary["TotalQuantity"].sum()
    return summary

# === Final KPI Comparison Table ===
kpi_tables = {}
kpi_tables["RL_Basic"] = compute_kpis(df_eval_basic).loc["Firm_A"]
kpi_tables["RL_Strategy"] = compute_kpis(df_eval_strategy).loc["Firm_A"]
kpi_tables["RL_Strategy+Decay"] = compute_kpis(df_eval_final).loc["Firm_A"]
kpi_tables["Rule_Based_B"] = compute_kpis(df_eval_final).loc["Firm_B"]
kpi_tables["Rule_Based_C"] = compute_kpis(df_eval_final).loc["Firm_C"]

kpi_comparison = pd.DataFrame(kpi_tables).T.round(2)

# === Show Results ===
print("\n KPI Comparison Across RL Versions and Baselines:")
print(kpi_comparison)

# Optional: Save to CSV
kpi_comparison.to_csv("kpi_comparison_results.csv")


#Competitive multi-agent RL environment
# Trend signal (with shocks and normalization)
np.random.seed(42)
weeks = list(range(1, 11))
base_trend = np.linspace(0.4, 0.7, len(weeks))
shock = np.zeros(len(weeks))
shock[3] = 0.5  # viral spike at week 4
shock[7:] = -0.3
noise = np.random.normal(0, 0.05, len(weeks))
trend_scores = MinMaxScaler().fit_transform((base_trend + shock + noise).reshape(-1, 1)).flatten()

# Bayesian trend update

def bayesian_trend_update(observed, prior_mean, prior_var, obs_var):
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    posterior_mean = (obs_var * prior_mean + prior_var * observed) / (prior_var + obs_var)
    return posterior_mean, posterior_var

# Softmax utility

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Strategy to price mapping

def get_price(strategy, week):
    if strategy == 0:
        return 10
    elif strategy == 1:
        return 8 if week >= 5 else 10
    elif strategy == 2:
        return 7 if week in [3, 6] else 10

# State representation
price_bins = [7, 8, 9, 10]
demand_bins = [0, 2000, 4000, 6000, 8000]
trend_bins = np.linspace(0, 1, 5)

def get_state(trend_score, last_price, ref_price, past_demand):
    t_bin = np.digitize(trend_score, trend_bins) - 1
    lp_bin = np.digitize(last_price, price_bins) - 1
    rp_bin = np.digitize(ref_price, price_bins) - 1
    d_bin = np.digitize(past_demand, demand_bins) - 1
    return (t_bin, lp_bin, rp_bin, d_bin)

# --- Multi-Agent Training ---
strategies = [0, 1, 2]  # static, markdown, promo

# Each firm has its own Q-table
Q_tables = {
    'Firm_A': defaultdict(lambda: np.zeros(len(strategies))),
    'Firm_B': defaultdict(lambda: np.zeros(len(strategies))),
    'Firm_C': defaultdict(lambda: np.zeros(len(strategies)))
}

episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 1.0
decay_rate = 0.995
epsilon_min = 0.05

reward_log = {firm: [] for firm in Q_tables}

for ep in range(episodes):
    priors = {firm: (0.5, 0.05) for firm in Q_tables}
    memories = {firm: {'last_price': 10, 'ref_price': 10, 'past_demand': 3000} for firm in Q_tables}
    episode_rewards = {firm: 0 for firm in Q_tables}

    for week_idx, week in enumerate(weeks):
        trend_obs = trend_scores[week_idx]
        trend_belief = {}
        for firm in Q_tables:
            mean, var = bayesian_trend_update(trend_obs, *priors[firm], obs_var=0.02)
            trend_belief[firm] = mean
            priors[firm] = (mean, var)

        firm_prices = {}
        states = {}
        actions = {}

        for firm in Q_tables:
            last = memories[firm]['last_price']
            ref = memories[firm]['ref_price']
            demand = memories[firm]['past_demand']
            state = get_state(trend_belief[firm], last, ref, demand)
            states[firm] = state

            if np.random.rand() < epsilon:
                action = np.random.choice(strategies)
            else:
                action = np.argmax(Q_tables[firm][state])

            actions[firm] = action
            firm_prices[firm] = get_price(action, week)

        # Compute market share
        firm_utils = {f: -p + trend_belief[f] for f, p in firm_prices.items()}
        shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_obs
        firm_names = list(firm_prices.keys())
        quantities = dict(zip(firm_names, shares * total_demand))

        for i, firm in enumerate(firm_names):
            price = firm_prices[firm]
            quantity = quantities[firm]
            profit = (price - 5) * quantity
            episode_rewards[firm] += profit

            # Q Update
            ref = memories[firm]['ref_price']
            next_state = get_state(trend_belief[firm], price, ref, quantity)
            Q_tables[firm][states[firm]][actions[firm]] += alpha * (
                profit + gamma * np.max(Q_tables[firm][next_state]) - Q_tables[firm][states[firm]][actions[firm]]
            )

            # Update memory
            memories[firm]['ref_price'] = memories[firm]['last_price']
            memories[firm]['last_price'] = price
            memories[firm]['past_demand'] = quantity

    for firm in Q_tables:
        reward_log[firm].append(episode_rewards[firm])

    epsilon = max(epsilon_min, epsilon * decay_rate)

# --- Evaluation Function ---
def evaluate_policy(Q_table):
    results = []
    memory = {'last_price': 10, 'ref_price': 10, 'past_demand': 3000}
    prior_mean, prior_var = 0.5, 0.05

    for week_idx, week in enumerate(weeks):
        observed = trend_scores[week_idx]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief

        state = get_state(trend_belief, memory['last_price'], memory['ref_price'], memory['past_demand'])
        strategy = np.argmax(Q_table[state])
        price = get_price(strategy, week)

        firm_prices = {
            'Firm_A': price,
            'Firm_B': get_price(np.argmax(Q_tables['Firm_B'][state]), week),
            'Firm_C': get_price(np.argmax(Q_tables['Firm_C'][state]), week)
        }

        firm_utils = {f: -p + trend_belief for f, p in firm_prices.items()}
        shares = softmax(np.array(list(firm_utils.values())))
        total_demand = 1500 + 500 * trend_belief
        firms = list(firm_prices.keys())
        quantities = dict(zip(firms, shares * total_demand))

        for f in firms:
            p, q = firm_prices[f], quantities[f]
            results.append({
                'Week': week, 'Firm': f, 'Price': p,
                'Quantity': q, 'Revenue': p * q, 'Profit': (p - 5) * q
            })

        memory['ref_price'] = memory['last_price']
        memory['last_price'] = price
        memory['past_demand'] = quantities['Firm_A']

    return pd.DataFrame(results)

# --- Evaluate all ---
df_eval_final = evaluate_policy(Q_tables['Firm_A'])

# --- KPI Comparison ---
def compute_kpis(df):
    summary = df.groupby("Firm").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Price": "mean"
    }).rename(columns={"Quantity": "TotalQuantity", "Price": "AvgPrice"})
    summary["ROI"] = summary["Profit"] / summary["Revenue"]
    summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / summary["TotalQuantity"].sum()
    return summary

print("\n Final Competitive RL Evaluation KPIs:")
print(compute_kpis(df_eval_final).round(2))

# Count strategy use per firm
strategy_counts = {firm: [] for firm in Q_tables}

for firm in Q_tables:
    q_table = Q_tables[firm]
    for state in q_table:
        strategy = np.argmax(q_table[state])
        strategy_counts[firm].append(strategy)

# Convert to DataFrame
strategy_df = pd.DataFrame({
    firm: pd.Series(vals).value_counts().sort_index()
    for firm, vals in strategy_counts.items()
}).T.fillna(0).astype(int)

strategy_df.columns = ['Static', 'Markdown', 'Promo']
print("\nStrategy Usage Frequency:")
print(strategy_df)

# Plot
strategy_df.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set2')
plt.title("Strategy Usage per Firm (based on Q-table)")
plt.xlabel("Firm")
plt.ylabel("Strategy Count in Q-Table")
plt.grid(True)
plt.tight_layout()
plt.show()

##Add Behavioral Segments to Simulation
# Parameters
trend_sensitive_ratio = 0.7  # 70% of users respond to trends; 30% ignore them
unit_cost = 5

# For plotting and results
results = []
last_prices = {}
weeks = list(range(1, 11))

# Trend score already created as trend_scores with shocks
prior_mean = 0.5
prior_var = 0.05
obs_var = 0.02

# Firm strategies
firms = {
    'Firm_A': {'strategy': 'static', 'base_price': 10},
    'Firm_B': {'strategy': 'markdown', 'base_price': 10, 'markdown_week': 5, 'markdown_price': 8},
    'Firm_C': {'strategy': 'promotion', 'base_price': 10, 'promo_weeks': [3, 6], 'promo_price': 7}
}

def get_firm_price(firm, week):
    strat = firm['strategy']
    if strat == 'static':
        return firm['base_price']
    elif strat == 'markdown':
        return firm['markdown_price'] if week >= firm['markdown_week'] else firm['base_price']
    elif strat == 'promotion':
        return firm['promo_price'] if week in firm['promo_weeks'] else firm['base_price']
    return firm['base_price']

for week_idx, week in enumerate(weeks):
    # Bayesian belief update
    observed = trend_scores[week_idx]
    trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, obs_var)
    prior_mean = trend_belief

    # Firm pricing and utilities
    firm_prices = {}
    trend_util = {}
    no_trend_util = {}

    for name, firm in firms.items():
        price = get_firm_price(firm, week)
        ref = last_prices.get(name, price)
        firm_prices[name] = price

        # Utility components
        trend_util[name] = -price + trend_belief
        no_trend_util[name] = -price  # ignores trend score

        last_prices[name] = price

    # Convert utilities to shares
    firms_list = list(firm_prices.keys())

    trend_shares = softmax([trend_util[f] for f in firms_list])
    no_trend_shares = softmax([no_trend_util[f] for f in firms_list])

    # Weighted avg of both consumer segments
    blended_shares = trend_sensitive_ratio * trend_shares + (1 - trend_sensitive_ratio) * no_trend_shares

    total_demand = 1500 + 500 * trend_belief

    for i, firm in enumerate(firms_list):
        share = blended_shares[i]
        price = firm_prices[firm]
        quantity = share * total_demand
        revenue = price * quantity
        profit = (price - unit_cost) * quantity

        results.append({
            'Week': week,
            'Firm': firm,
            'Price': price,
            'TrendScore_Belief': trend_belief,
            'MarketShare': share,
            'PredictedQuantity': quantity,
            'Revenue': revenue,
            'Profit': profit
        })

# Convert to DataFrame
df_behavioral = pd.DataFrame(results)

# KPI Summary
def compute_kpis(df):
    summary = df.groupby("Firm").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "PredictedQuantity": "sum",
        "Price": "mean"
    }).rename(columns={
        "PredictedQuantity": "TotalQuantity",
        "Price": "AvgPrice"
    })
    summary["ROI"] = summary["Profit"] / summary["Revenue"]
    summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / summary["TotalQuantity"].sum()
    return summary.round(2)

print("\n KPI Summary with Trend-Insensitive Consumers:")
print(compute_kpis(df_behavioral))

# --- Safe check for quantity column ---
qty_col = 'PredictedQuantity' if 'PredictedQuantity' in df_behavioral.columns else 'Quantity'

# --- Add Conversion Rate Column ---
df_behavioral['TotalDemand'] = 1500 + 500 * df_behavioral['TrendScore_Belief']
df_behavioral['ConversionRate'] = df_behavioral[qty_col] / df_behavioral['TotalDemand']

# --- Compute average Conversion Rate per Firm ---
conversion_rate_summary = df_behavioral.groupby("Firm")["ConversionRate"].mean().reset_index()
conversion_rate_summary.columns = ["Firm", "AvgConversionRate"]

# --- Define proper KPI computation function for firm-level analysis ---
def compute_kpis_firm_level(df):
    quantity_col = 'PredictedQuantity' if 'PredictedQuantity' in df.columns else 'Quantity'
    summary = df.groupby("Firm").agg({
        "Revenue": "sum",
        "Profit": "sum",
        quantity_col: "sum",
        "Price": "mean"
    }).rename(columns={
        quantity_col: "TotalQuantity",
        "Price": "AvgPrice"
    })

    summary["ROI"] = summary["Profit"] / summary["Revenue"]
    summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / summary["TotalQuantity"].sum()
    return summary.round(2)

# --- Compute KPI summary and merge with conversion rate ---
kpi_summary = compute_kpis_firm_level(df_behavioral).reset_index()
kpi_summary = pd.merge(kpi_summary, conversion_rate_summary, on="Firm")

# --- Print final KPI summary ---
print("\n KPI Summary with Conversion Rates:")
print(kpi_summary.round(3))



#Sensitivity analysis 
# --- Assumed definitions ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def bayesian_trend_update(observed, prior_mean, prior_var, obs_var):
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    posterior_mean = (obs_var * prior_mean + prior_var * observed) / (prior_var + obs_var)
    return posterior_mean, posterior_var

def get_price(strategy, week):
    if strategy == 0: return 10
    elif strategy == 1: return 8 if week >= 5 else 10
    elif strategy == 2: return 7 if week in [3, 6] else 10

# --- Trend scores with viral spike + noise ---
np.random.seed(42)
weeks = list(range(1, 11))
base_trend = np.linspace(0.4, 0.7, len(weeks))
shock = np.zeros(len(weeks)); shock[3] = 0.5; shock[7:] = -0.3
noise = np.random.normal(0, 0.05, len(weeks))
trend_scores = MinMaxScaler().fit_transform((base_trend + shock + noise).reshape(-1, 1)).flatten()

# --- Compute Rolling Trend Volatility Index (TVI) ---
trend_df = pd.DataFrame({
    'Week': list(range(1, len(trend_scores) + 1)),
    'TrendScore': trend_scores
})

# Rolling 3-week standard deviation of trend score
trend_df['TVI'] = trend_df['TrendScore'].rolling(window=3, min_periods=1).std()

# Optional: plot volatility
plt.plot(trend_df['Week'], trend_df['TVI'], marker='o', linestyle='--', color='crimson')
plt.title("Trend Volatility Index (TVI)")
plt.xlabel("Week")
plt.ylabel("Volatility (Rolling Std Dev)")
plt.grid(True)
plt.show()


# --- Sensitivity levels to test ---
sensitivity_levels = [0.3, 0.5, 0.7, 0.9, 1.0]  # 30% to 100% trend-responsive

# --- Storage for all results ---
kpi_records = []

for trend_ratio in sensitivity_levels:
    results = []
    last_prices = {'Firm_A': 10, 'Firm_B': 10, 'Firm_C': 10}
    ref_prices = {'Firm_A': 10, 'Firm_B': 10, 'Firm_C': 10}
    prior_mean, prior_var = 0.5, 0.05

    for week_idx, week in enumerate(weeks):
        observed = trend_scores[week_idx]
        trend_belief, prior_var = bayesian_trend_update(observed, prior_mean, prior_var, 0.02)
        prior_mean = trend_belief

        # Prices: Static, Markdown, Promo
        firm_strategies = {'Firm_A': 0, 'Firm_B': 1, 'Firm_C': 2}
        firm_prices = {firm: get_price(strategy, week) for firm, strategy in firm_strategies.items()}
        firm_utils = {firm: -price + trend_belief for firm, price in firm_prices.items()}

        # Adjust for behavior: only trend_ratio% respond to utility, rest act randomly
        utility_shares = softmax(np.array(list(firm_utils.values())))
        random_shares = np.ones_like(utility_shares) / len(utility_shares)
        final_shares = trend_ratio * utility_shares + (1 - trend_ratio) * random_shares

        total_demand = 1500 + 500 * trend_belief
        firm_list = list(firm_prices.keys())
        firm_quantities = dict(zip(firm_list, final_shares * total_demand))

        for firm in firm_list:
            price = firm_prices[firm]
            quantity = firm_quantities[firm]
            revenue = price * quantity
            profit = (price - 5) * quantity
            results.append({
                'Week': week,
                'Firm': firm,
                'Price': price,
                'Quantity': quantity,
                'Revenue': revenue,
                'Profit': profit,
                'TrendRatio': trend_ratio
            })

            # update prices
            ref_prices[firm] = last_prices[firm]
            last_prices[firm] = price

    df = pd.DataFrame(results)
    summary = df.groupby("Firm").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Price": "mean"
    }).rename(columns={"Quantity": "TotalQuantity", "Price": "AvgPrice"})
    summary["ROI"] = summary["Profit"] / summary["Revenue"]
    summary["MarketShare(%)"] = 100 * summary["TotalQuantity"] / summary["TotalQuantity"].sum()
    summary["TrendRatio"] = trend_ratio
    kpi_records.append(summary.reset_index())

# --- Combine all KPI summaries ---
kpi_df = pd.concat(kpi_records).reset_index(drop=True)
print("\n KPI Impact by Trend-Sensitivity Level:")
print(kpi_df.round(2))

# Plot Market Share vs TrendRatio
plt.figure(figsize=(10,6))
sns.lineplot(data=kpi_df, x="TrendRatio", y="MarketShare(%)", hue="Firm", marker="o")
plt.title("Market Share vs Trend Sensitivity")
plt.ylabel("Market Share (%)")
plt.grid(True)
plt.show()


# -----------------------------
# TREND DATA SECTION (REAL vs SYNTHETIC)
# -----------------------------

# Set this flag: True for Google Trends data, False for synthetic trend
USE_REAL_TRENDS = True  # Toggle this

# -----------------------------
# CASE 1: USE REAL GOOGLE TRENDS DATA
# -----------------------------

if USE_REAL_TRENDS:
    print("Trying to load REAL trend data from Google Trends...")

    keyword = "TikTok"
    retries = 3
    delay = 10  # seconds

    for attempt in range(retries):
        try:
            pytrends = TrendReq(hl='en-UK', tz=360)
            pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='GB', gprop='')
            trend_df = pytrends.interest_over_time().reset_index()

            trend_df['RawTrend'] = trend_df[keyword]
            trend_df['TrendScore'] = MinMaxScaler().fit_transform(trend_df[['RawTrend']])
            trend_df = trend_df[['date', 'RawTrend', 'TrendScore']].copy()

            print(trend_df.head())
            trend_df.to_csv("google_trend_general.csv", index=False)

            trend_scores = trend_df['TrendScore'].values[:10]
            weeks = list(range(1, len(trend_scores) + 1))
            break  # success, exit loop

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                print("Switching to synthetic data instead.")
                USE_REAL_TRENDS = False

# -----------------------------
# CASE 2: SYNTHETIC TREND DATA
# -----------------------------

if not USE_REAL_TRENDS:
    print("Using SYNTHETIC trend data (spike + decay)...")
    np.random.seed(42)
    weeks = list(range(1, 11))
    base_trend = np.linspace(0.4, 0.7, len(weeks))
    shock = np.zeros(len(weeks))
    shock[3] = 0.5
    shock[7:] = -0.3
    noise = np.random.normal(0, 0.05, len(weeks))
    synthetic_trend = base_trend + shock + noise
    synthetic_trend = np.clip(synthetic_trend, 0, 1)
    trend_scores = MinMaxScaler().fit_transform(synthetic_trend.reshape(-1, 1)).flatten()

print("\n=== FINAL Trend Scores to be used ===")
for i, score in enumerate(trend_scores):
    print(f"Week {i+1}: {score:.3f}")

num_firms = 3
strategies = ['markdown', 'promotion', 'none']
results = []

for week, trend_score in enumerate(trend_scores, start=1):
    for firm_id in range(num_firms):
        # Sample strategy
        strategy = random.choice(strategies)  # or RL agent decision here

        # Calculate price adjustment
        price = base_price
        if strategy == 'markdown':
            price *= 0.8
        elif strategy == 'promotion':
            price *= 0.9

        # Simulate demand (simple model: trend score × price sensitivity)
        demand = (trend_score * 100) * np.exp(-price / 10)

        # Save results
        results.append({
            'Week': week,
            'FirmID': firm_id,
            'Strategy': strategy,
            'Price': price,
            'UnitsSold': demand,
            'Revenue': price * demand,
            'TrendScore': trend_score
        })

results_df = pd.DataFrame(results)

def compute_kpis(df):
    kpis = df.groupby('Strategy').agg({
        'Revenue': 'sum',
        'UnitsSold': 'sum',
        'FirmID': 'nunique'
    }).reset_index()
    return kpis

kpi_df = compute_kpis(results_df)
print(kpi_df)
kpi_df.to_csv("kpi_summary.csv", index=False)

sns.lineplot(data=results_df, x='Week', y='Revenue', hue='Strategy')
plt.title("Revenue Over Time by Strategy")
plt.show()

# ---------------------------------------------
# STEP 1: Add Profit Column to Each Row
# ---------------------------------------------
UNIT_COST = 5  # Assumed cost per unit
results_df['Profit'] = (results_df['Price'] - UNIT_COST) * results_df['UnitsSold']

# ---------------------------------------------
# STEP 2: Recompute KPIs with Profit, ROI, Market Share
# ---------------------------------------------
def compute_enhanced_kpis(df):
    kpis = df.groupby('Strategy').agg({
        'Revenue': 'sum',
        'UnitsSold': 'sum',
        'Profit': 'sum',
        'FirmID': 'nunique'
    }).rename(columns={'FirmID': 'NumFirms'}).reset_index()

    kpis['ROI'] = kpis['Profit'] / kpis['Revenue']
    total_units = kpis['UnitsSold'].sum()
    kpis['MarketShare(%)'] = 100 * kpis['UnitsSold'] / total_units

    return kpis.round(2)

kpi_df = compute_enhanced_kpis(results_df)
print("\n=== Enhanced KPI Summary ===")
print(kpi_df)

# Save updated results
results_df.to_csv("simulation_results.csv", index=False)
kpi_df.to_csv("enhanced_kpi_summary.csv", index=False)

# ---------------------------------------------
# STEP 3: Visualize More Insights
# ---------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Revenue over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='Revenue', hue='Strategy', marker='o')
plt.title("Revenue Over Time by Strategy")
plt.ylabel("Revenue")
plt.grid(True)
plt.tight_layout()
plt.show()

#  Units Sold over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='UnitsSold', hue='Strategy', marker='s')
plt.title(" Units Sold Over Time by Strategy")
plt.ylabel("Units Sold")
plt.grid(True)
plt.tight_layout()
plt.show()

#  Profit over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='Profit', hue='Strategy', marker='d')
plt.title("Profit Over Time by Strategy")
plt.ylabel("Profit")
plt.grid(True)
plt.tight_layout()
plt.show()

#  Final KPI Bar Charts
kpi_df.set_index('Strategy')[['Revenue', 'Profit', 'UnitsSold']].plot(kind='bar', figsize=(10, 5))
plt.title(" Strategy Performance Summary (Total Values)")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Market Share
plt.figure(figsize=(8, 4))
sns.barplot(data=kpi_df, x='Strategy', y='MarketShare(%)', palette='muted')
plt.title(" Market Share by Strategy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Trend Volatility Index ---
# This captures how unstable or noisy the trend is over time (used later in modeling)

# Calculate volatility using rolling 3-day standard deviation
trend_df['TrendVolatility'] = trend_df['TrendScore'].rolling(window=3).std().fillna(0)

# Optional: Print to verify
print("\nTrend Scores with Volatility Index:")
print(trend_df[['date', 'TrendScore', 'TrendVolatility']].head(10))

# Optional: Save updated trend data to CSV
trend_df.to_csv("google_trend_general.csv", index=False)

for week_idx, obs in enumerate(trend_scores):
    posterior_mean, posterior_var = bayesian_trend_update(
        observed=obs,
        prior_mean=prior_mean,
        prior_var=prior_var,
        obs_var=0.02  # <-- replace this
    )

    obs_var = max(0.01, trend_df.loc[week_idx, 'TrendVolatility'])  # cap minimum for stability

belief_means = []
belief_vars = []

for week_idx, obs in enumerate(trend_scores):
    # Use volatility as a proxy for observation uncertainty
    obs_var = max(0.01, trend_df.loc[week_idx, 'TrendVolatility'])  # Ensure it's not zero

    posterior_mean, posterior_var = bayesian_trend_update(
        observed=obs,
        prior_mean=prior_mean,
        prior_var=prior_var,
        obs_var=obs_var
    )

    belief_means.append(posterior_mean)
    belief_vars.append(posterior_var)

    prior_mean = posterior_mean
    prior_var = posterior_var

plt.figure(figsize=(10,5))
plt.plot(weeks, trend_scores, label="Observed TrendScore", linestyle='--', marker='o')
plt.plot(weeks, belief_means, label="Bayesian Belief", linestyle='-', marker='s')
plt.fill_between(
    weeks,
    np.array(belief_means) - 1.96 * np.sqrt(belief_vars),
    np.array(belief_means) + 1.96 * np.sqrt(belief_vars),
    color='orange', alpha=0.3, label="95% Confidence"
)
plt.title("Bayesian Trend Belief Update Over Time")
plt.xlabel("Week")
plt.ylabel("TrendScore")
plt.legend()
plt.grid(True)
plt.show()


# --- Assumed Inputs ---
num_firms = 3
base_price = 10
UNIT_COST = 5

results = []

# === SIMULATION LOOP ===
for week_idx, trend_score in enumerate(belief_means):
    week = week_idx + 1
    volatility = trend_df.loc[week_idx, 'TrendVolatility']

    for firm_id in range(num_firms):
        # Volatility-based strategy logic
        if volatility < 0.02:
            strategy = 'promotion'
        elif volatility > 0.05:
            strategy = 'markdown'
        else:
            strategy = random.choice(['promotion', 'none'])

        # Adjust price
        price = base_price
        if strategy == 'markdown':
            price *= 0.8
        elif strategy == 'promotion':
            price *= 0.9

        # Simulate demand using Bayesian trend score
        demand = (trend_score * 100) * np.exp(-price / 10)

        # Store results
        results.append({
            'Week': week,
            'FirmID': firm_id,
            'Strategy': strategy,
            'Price': price,
            'UnitsSold': demand,
            'Revenue': price * demand,
            'Profit': (price - UNIT_COST) * demand,
            'TrendScore': trend_score,
            'TrendVolatility': volatility
        })

# === Create DataFrame ===
results_df = pd.DataFrame(results)

# === Compute Enhanced KPIs ===
def compute_enhanced_kpis(df):
    kpis = df.groupby('Strategy').agg({
        'Revenue': 'sum',
        'UnitsSold': 'sum',
        'Profit': 'sum',
        'FirmID': 'nunique'
    }).rename(columns={'FirmID': 'NumFirms'}).reset_index()

    kpis['ROI'] = kpis['Profit'] / kpis['Revenue']
    total_units = kpis['UnitsSold'].sum()
    kpis['MarketShare(%)'] = 100 * kpis['UnitsSold'] / total_units

    return kpis.round(2)

kpi_df = compute_enhanced_kpis(results_df)

# === Show KPI Summary ===
print("\n=== KPI Summary (Volatility-Aware Strategy) ===")
print(kpi_df)

# === Save Outputs ===
results_df.to_csv("simulation_results_volatility.csv", index=False)
kpi_df.to_csv("kpi_summary_volatility.csv", index=False)

# === Plotting ===
sns.set(style="whitegrid")

# Revenue Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='Revenue', hue='Strategy', marker='o')
plt.title("Revenue Over Time by Strategy (Volatility-Aware)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Units Sold Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='UnitsSold', hue='Strategy', marker='s')
plt.title("Units Sold Over Time by Strategy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Profit Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x='Week', y='Profit', hue='Strategy', marker='D')
plt.title("Profit Over Time by Strategy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Plot: KPI Summary
kpi_df.set_index('Strategy')[['Revenue', 'Profit', 'UnitsSold']].plot(kind='bar', figsize=(10, 5))
plt.title("Strategy Performance Summary (Volatility-Aware)")
plt.ylabel("Total Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Market Share Plot
plt.figure(figsize=(8, 4))
sns.barplot(data=kpi_df, x='Strategy', y='MarketShare(%)', palette='muted')
plt.title("Market Share by Strategy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

results_random = []

for week_idx, trend_score in enumerate(belief_means):
    week = week_idx + 1
    volatility = trend_df.loc[week_idx, 'TrendVolatility']

    for firm_id in range(num_firms):
        # RANDOM strategy: baseline for comparison
        strategy = random.choice(['promotion', 'markdown', 'none'])

        price = base_price
        if strategy == 'markdown':
            price *= 0.8
        elif strategy == 'promotion':
            price *= 0.9

        demand = (trend_score * 100) * np.exp(-price / 10)

        results_random.append({
            'Week': week,
            'FirmID': firm_id,
            'Strategy': strategy,
            'Price': price,
            'UnitsSold': demand,
            'Revenue': price * demand,
            'Profit': (price - UNIT_COST) * demand,
            'TrendScore': trend_score,
            'TrendVolatility': volatility,
            'StrategyType': 'Random'
        })

results_df_random = pd.DataFrame(results_random)

kpi_df_random = compute_enhanced_kpis(results_df_random)
kpi_df_random['StrategyType'] = 'Random'

# Add label to previous run
results_df['StrategyType'] = 'VolatilityAware'
kpi_df['StrategyType'] = 'VolatilityAware'

# Combine
all_kpis = pd.concat([kpi_df, kpi_df_random], ignore_index=True)
all_results = pd.concat([results_df, results_df_random], ignore_index=True)

# Save for report
all_kpis.to_csv("kpi_comparison.csv", index=False)
all_results.to_csv("all_simulation_results.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=all_kpis, x='Strategy', y='Profit', hue='StrategyType')
plt.title("Profit by Strategy Type (Volatility vs Random)")
plt.ylabel("Total Profit")
plt.grid(True)
plt.tight_layout()
plt.show()


#Robustness Checks 
# Monte Carlo 
def run_single_simulation(trend_scores, weeks, num_firms, base_price=10, unit_cost=5):
    results = []
    for week, trend_score in zip(weeks, trend_scores):
        for firm_id in range(num_firms):
            strategy = random.choice(['markdown', 'promotion', 'none'])
            price = base_price
            if strategy == 'markdown':
                price *= 0.8
            elif strategy == 'promotion':
                price *= 0.9
            demand = (trend_score * 100) * np.exp(-price / 10)
            revenue = price * demand
            profit = (price - unit_cost) * demand
            results.append({
                'Week': week,
                'FirmID': firm_id,
                'Strategy': strategy,
                'Price': price,
                'UnitsSold': demand,
                'Revenue': revenue,
                'Profit': profit,
                'TrendScore': trend_score
            })
    df = pd.DataFrame(results)
    kpi = df.groupby('Strategy')[['Revenue', 'Profit', 'UnitsSold']].sum().reset_index()
    kpi['ROI'] = kpi['Profit'] / kpi['Revenue']
    kpi['Run'] = None  # Will be added later
    return kpi

def run_monte_carlo_simulation(trend_scores, weeks, num_firms=3, runs=100):
    all_kpis = []
    for i in range(runs):
        kpi_df = run_single_simulation(trend_scores, weeks, num_firms)
        kpi_df['Run'] = i + 1
        all_kpis.append(kpi_df)
    combined_kpis = pd.concat(all_kpis, ignore_index=True)
    summary = combined_kpis.groupby('Strategy').agg({
        'Revenue': ['mean', 'std'],
        'Profit': ['mean', 'std'],
        'UnitsSold': ['mean', 'std'],
        'ROI': ['mean', 'std']
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.round(2), combined_kpis

summary_stats, all_runs_df = run_monte_carlo_simulation(trend_scores, weeks, num_firms=3, runs=100)

print("\n=== Monte Carlo KPI Summary (100 runs) ===")
print(summary_stats)

# Optional: Save to CSV
summary_stats.to_csv("monte_carlo_summary.csv")
all_runs_df.to_csv("monte_carlo_all_runs.csv", index=False)


#Confidence interval 
def add_confidence_intervals(df, metric, group='Strategy'):
    grouped = df.groupby(group)[metric]
    mean = grouped.mean()
    sem = grouped.sem()
    ci95 = sem * stats.t.ppf((1 + 0.95) / 2., grouped.count() - 1)
    return pd.DataFrame({
        'mean': mean,
        'ci95': ci95
    }).round(2)

all_runs_df['Revenue_per_Unit'] = all_runs_df['Revenue'] / all_runs_df['UnitsSold']
top_strat_freq = all_runs_df.groupby(['Run']).apply(
    lambda x: x.loc[x['Revenue'].idxmax(), 'Strategy']
).value_counts()
print(top_strat_freq)


# Vary Price elasticity 
def run_elasticity_sensitivity(trend_scores, weeks, num_firms=3, k_values=[6, 8, 10, 12, 14], runs_per_k=50):
    all_results = []

    for k in k_values:
        print(f"Running elasticity simulation for k = {k}...")
        for run in range(runs_per_k):
            results = []
            for week, trend_score in zip(weeks, trend_scores):
                for firm_id in range(num_firms):
                    strategy = random.choice(['markdown', 'promotion', 'none'])

                    price = base_price
                    if strategy == 'markdown':
                        price *= 0.8
                    elif strategy == 'promotion':
                        price *= 0.9

                    # Demand based on current elasticity value
                    demand = (trend_score * 100) * np.exp(-price / k)
                    revenue = price * demand
                    profit = (price - 5) * demand

                    results.append({
                        'ElasticityK': k,
                        'Run': run + 1,
                        'Week': week,
                        'FirmID': firm_id,
                        'Strategy': strategy,
                        'Price': price,
                        'UnitsSold': demand,
                        'Revenue': revenue,
                        'Profit': profit,
                        'TrendScore': trend_score
                    })

            df = pd.DataFrame(results)
            kpi = df.groupby('Strategy')[['Revenue', 'Profit', 'UnitsSold']].sum().reset_index()
            kpi['ROI'] = kpi['Profit'] / kpi['Revenue']
            kpi['ElasticityK'] = k
            kpi['Run'] = run + 1
            all_results.append(kpi)

    final_df = pd.concat(all_results, ignore_index=True)
    summary_df = final_df.groupby(['ElasticityK', 'Strategy']).agg({
        'Revenue': ['mean', 'std'],
        'Profit': ['mean', 'std'],
        'UnitsSold': ['mean', 'std'],
        'ROI': ['mean', 'std']
    }).round(2)

    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    return summary_df, final_df

elasticity_summary, elasticity_all_runs = run_elasticity_sensitivity(
    trend_scores,
    weeks,
    num_firms=3,
    k_values=[6, 8, 10, 12, 14],
    runs_per_k=50
)

# View summary
print("\n===  Price Elasticity Sensitivity Summary ===")
print(elasticity_summary)

# Save if needed
elasticity_summary.to_csv("elasticity_kpi_summary.csv")
elasticity_all_runs.to_csv("elasticity_all_runs.csv", index=False)

# Flatten index
elasticity_plot = elasticity_summary.reset_index()

# Revenue Plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=elasticity_plot, x='ElasticityK', y='Revenue_mean', hue='Strategy', marker='o')
plt.title("Revenue vs Price Elasticity (k)")
plt.grid(True)
plt.show()

# Profit Plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=elasticity_plot, x='ElasticityK', y='Profit_mean', hue='Strategy', marker='s')
plt.title("Profit vs Price Elasticity (k)")
plt.grid(True)
plt.show()

# ROI Plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=elasticity_plot, x='ElasticityK', y='ROI_mean', hue='Strategy', marker='d')
plt.title("ROI vs Price Elasticity (k)")
plt.grid(True)
plt.show()

# Baseline at k = 10
baseline = elasticity_summary.loc[10]

relative_change = (
    elasticity_summary
    .reset_index()
    .merge(baseline.reset_index(), on='Strategy', suffixes=('', '_baseline'))
)

relative_change['Revenue_pct_change'] = (
    (relative_change['Revenue_mean'] - relative_change['Revenue_mean_baseline'])
    / relative_change['Revenue_mean_baseline']
) * 100


sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=elasticity_all_runs,
    x='ElasticityK',
    y='Revenue',
    hue='Strategy',
    estimator='mean',
    errorbar='sd',
    marker='o'
)
plt.title("Revenue vs Elasticity with Std Dev")
plt.grid(True)
plt.tight_layout()
plt.show()

best_by_k = elasticity_all_runs.groupby(['ElasticityK', 'Run']).apply(
    lambda x: x.loc[x['Revenue'].idxmax(), 'Strategy']
).reset_index(name='BestStrategy')

# Count how often each strategy wins
print(best_by_k.groupby(['ElasticityK', 'BestStrategy']).size().unstack(fill_value=0))

#Consumer Conversion check 
# --- Consumer Conversion Rate in Shared Demand Simulation ---

# Assuming 'results_df' contains shared demand simulation results:
# If you're rerunning the simulation, define this first
results = []
last_prices = {}
unit_cost = 5
weeks = list(range(1, 11))  # adjust if using a different week range

# Sample trend (or reuse trend_scores from your earlier simulation)
trend_scores = MinMaxScaler().fit_transform(np.linspace(0.4, 0.7, len(weeks)).reshape(-1, 1)).flatten()

for week_idx, week in enumerate(weeks):
    trend_score = trend_scores[week_idx]

    firm_prices = {}
    firm_utils = {}
    for firm_name, firm in firms.items():
        base_price = firm['base_price']
        if firm['strategy'] == 'static':
            price = base_price
        elif firm['strategy'] == 'markdown':
            price = firm['markdown_price'] if week >= firm['markdown_week'] else base_price
        elif firm['strategy'] == 'promotion':
            price = firm['promo_price'] if week in firm['promo_weeks'] else base_price
        else:
            price = base_price

        ref_price = base_price if week == 1 else last_prices.get(firm_name, base_price)
        price_change = price - ref_price

        alpha = 1.0
        beta = 1.0
        utility = -alpha * price + beta * trend_score

        firm_prices[firm_name] = {
            'price': price,
            'ref_price': ref_price,
            'price_change': price_change
        }
        firm_utils[firm_name] = utility

    utilities = np.array(list(firm_utils.values()))
    market_shares = softmax(utilities)
    total_demand = 1500 + 500 * trend_score

    for i, (firm_name, firm) in enumerate(firms.items()):
        share = market_shares[i]
        quantity = total_demand * share
        price = firm_prices[firm_name]['price']
        revenue = price * quantity
        profit = (price - unit_cost) * quantity

        # --- NEW: Calculate Conversion Rate ---
        conversion_rate = quantity / total_demand

        results.append({
            'Week': week,
            'Firm': firm_name,
            'Price': price,
            'RefPrice': firm_prices[firm_name]['ref_price'],
            'TrendScore': trend_score,
            'MarketShare': share,
            'PredictedQuantity': quantity,
            'Revenue': revenue,
            'Profit': profit,
            'ConversionRate': conversion_rate  # New metric
        })

        last_prices[firm_name] = price

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df

# --- Summary: Average Conversion Rate per Firm ---
conversion_summary = results_df.groupby("Firm")["ConversionRate"].mean().round(4)
print("\nAverage Conversion Rate by Firm:")
print(conversion_summary)

# --- Optional: Plot Conversion Rate Over Time ---
plt.figure(figsize=(8, 5))
for firm in results_df['Firm'].unique():
    firm_data = results_df[results_df['Firm'] == firm]
    plt.plot(firm_data['Week'], firm_data['ConversionRate'], marker='o', label=firm)

plt.title("Conversion Rate Over Time by Firm")
plt.xlabel("Week")
plt.ylabel("Conversion Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# RANDOM TREND SHOCK TEST
# ==========================

# Simulate weeks
weeks = list(range(1, 11))
np.random.seed(101)

# Step 1: Create base trend
base_trend = np.linspace(0.4, 0.6, len(weeks))

# Step 2: Inject false-positive spike (week 3), decay afterwards
shock = np.zeros(len(weeks))
shock[2] = 0.5     # false spike at week 3
shock[3:6] = -0.3  # downward correction

# Step 3: Add noise
noise = np.random.normal(0, 0.03, len(weeks))

# Step 4: Combine and normalize
trend_series = np.clip(base_trend + shock + noise, 0, 1)
trend_scores = MinMaxScaler().fit_transform(trend_series.reshape(-1, 1)).flatten()

# Plot the trend
plt.plot(weeks, trend_scores, marker='o', linestyle='--', color='red')
plt.title("Trend Shock Simulation (False Spike at Week 3)")
plt.xlabel("Week")
plt.ylabel("Trend Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Define Firm Strategies (can skip if already defined) ===
firms = {
    'Firm_A': {'strategy': 'static', 'base_price': 10},
    'Firm_B': {'strategy': 'markdown', 'base_price': 10, 'markdown_week': 5, 'markdown_price': 8},
    'Firm_C': {'strategy': 'promotion', 'base_price': 10, 'promo_weeks': [3, 6], 'promo_price': 7}
}

# === Define softmax helper ===
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# === Function: Generate Trend Scores ===
def generate_trend_scores(mode="baseline", seed=42):
    np.random.seed(seed)
    weeks = list(range(1, 11))
    base_trend = np.linspace(0.4, 0.6, len(weeks))

    if mode == "shock":
        shock = np.zeros(len(weeks))
        shock[2] = 0.5     # false viral spike at week 3
        shock[3:6] = -0.3  # crash after spike
        noise = np.random.normal(0, 0.03, len(weeks))
        combined = np.clip(base_trend + shock + noise, 0, 1)
    else:  # baseline
        noise = np.random.normal(0, 0.03, len(weeks))
        combined = np.clip(base_trend + noise, 0, 1)

    return MinMaxScaler().fit_transform(combined.reshape(-1, 1)).flatten()

# === Function: Run Shared Demand Simulation ===
def run_shared_simulation(trend_scores, label=""):
    results = []
    last_prices = {}
    unit_cost = 5
    weeks = list(range(1, 11))

    for week_idx, week in enumerate(weeks):
        trend_score = trend_scores[week_idx]
        firm_prices = {}
        firm_utils = {}

        for firm_name, firm in firms.items():
            base_price = firm['base_price']
            if firm['strategy'] == 'static':
                price = base_price
            elif firm['strategy'] == 'markdown':
                price = firm['markdown_price'] if week >= firm['markdown_week'] else base_price
            elif firm['strategy'] == 'promotion':
                price = firm['promo_price'] if week in firm['promo_weeks'] else base_price
            else:
                price = base_price

            ref_price = base_price if week == 1 else last_prices.get(firm_name, base_price)
            price_change = price - ref_price

            alpha, beta = 1.0, 1.0
            utility = -alpha * price + beta * trend_score

            firm_prices[firm_name] = {
                'price': price,
                'ref_price': ref_price,
                'price_change': price_change
            }
            firm_utils[firm_name] = utility

        utilities = np.array(list(firm_utils.values()))
        market_shares = softmax(utilities)
        total_demand = 1500 + 500 * trend_score

        for i, (firm_name, firm) in enumerate(firms.items()):
            share = market_shares[i]
            quantity = total_demand * share
            price = firm_prices[firm_name]['price']
            revenue = price * quantity
            profit = (price - unit_cost) * quantity
            conversion_rate = quantity / total_demand  # NEW METRIC

            results.append({
                'Week': week,
                'Firm': firm_name,
                'Price': price,
                'TrendScore': trend_score,
                'MarketShare': share,
                'PredictedQuantity': quantity,
                'Revenue': revenue,
                'Profit': profit,
                'ConversionRate': conversion_rate,
                'Scenario': label
            })

            last_prices[firm_name] = price

    return pd.DataFrame(results)

# === Run Simulations ===
baseline_trend = generate_trend_scores("baseline")
shock_trend = generate_trend_scores("shock")

df_baseline = run_shared_simulation(baseline_trend, label="Baseline")
df_shock = run_shared_simulation(shock_trend, label="Shock")

# === Merge for Comparison ===
df_combined = pd.concat([df_baseline, df_shock])

# === Summary KPI Table ===
summary = df_combined.groupby(["Scenario", "Firm"]).agg({
    "Revenue": "sum",
    "Profit": "sum",
    "PredictedQuantity": "sum",
    "ConversionRate": "mean"
}).rename(columns={"PredictedQuantity": "TotalQuantity"}).round(2)

print("\n KPI Comparison: Baseline vs. Shock Trend")
print(summary)

# === Optional Plot: Conversion Rate Over Time ===
plt.figure(figsize=(9, 5))
for scenario in df_combined['Scenario'].unique():
    for firm in df_combined['Firm'].unique():
        subset = df_combined[(df_combined['Scenario'] == scenario) & (df_combined['Firm'] == firm)]
        plt.plot(subset['Week'], subset['ConversionRate'], marker='o', label=f"{firm} ({scenario})")

plt.title("Conversion Rate Over Time — Baseline vs. Trend Shock")
plt.xlabel("Week")
plt.ylabel("Conversion Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Save Monte Carlo Simulation Results ===
summary_stats.to_csv("monte_carlo_summary.csv", index=False)
all_runs_df.to_csv("monte_carlo_all_runs.csv", index=False)

# Save confidence intervals for each KPI
for metric in ['Revenue', 'Profit', 'UnitsSold', 'ROI']:
    ci_df = add_confidence_intervals(all_runs_df, metric)
    ci_df.to_csv(f"{metric.lower()}_confidence_interval.csv")

# Save top-performing strategy frequency
top_strat_freq.to_csv("top_strategy_frequency.csv")

# === Save Elasticity Sensitivity Results ===
elasticity_summary.to_csv("elasticity_kpi_summary.csv")
elasticity_all_runs.to_csv("elasticity_all_runs.csv", index=False)

# Best strategy counts by elasticity level
best_by_k_summary = best_by_k.groupby(['ElasticityK', 'BestStrategy']).size().unstack(fill_value=0)
best_by_k_summary.to_csv("best_strategy_by_elasticity.csv")

# === Save Shared Demand Results with Conversion Rate ===
results_df.to_csv("shared_demand_conversion_results.csv", index=False)
conversion_summary.to_csv("average_conversion_by_firm.csv")

# === Save Trend Shock Simulation Results ===
df_combined.to_csv("trend_shock_simulation_all_runs.csv", index=False)
summary.to_csv("trend_shock_kpi_summary.csv")
