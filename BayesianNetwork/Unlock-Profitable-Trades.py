# Link : https://global-fxs.com/unlock-profitable-trades-with-ai/
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
pd.options.mode.chained_assignment = None

# Load your OHLC data
data = pd.read_csv('XAUUSD_M1_2.csv')
# Conver the Datetime to datetime object
data['Date'] = pd.to_datetime(data['Date'])
# Split the hour for future
data['hour'] = data['Date'].dt.hour
# Calculate the 50-period Simple Moving Average
data['SMA50'] = data['Close'].rolling(window=10).mean()
# Generate signals when close price crosses over or under SMA50
data['Signal'] = 0
data['Signal'][10:] = np.where(data['Close'][10:] > data['SMA50'][10:], 1, -1)
data['Position'] = data['Signal'].diff()

print(data.tail())
# Define minimum profit to consider a trade profitable
min_profit = 0.0001  # Adjust as needed (e.g., 1% profit)

# Initialize the Profit column
data['Profit'] = np.nan

# Calculate profit for each signal
for i in range(len(data)):
    if data['Position'].iloc[i] == 2:  # Buy signal
        entry_price = data['Close'].iloc[i]
        for j in range(i + 1, len(data)):
            price_change = (data['Close'].iloc[j] - entry_price) / entry_price
            if price_change >= min_profit or price_change <= -min_profit:
                data['Profit'].iloc[i] = price_change
                break
    elif data['Position'].iloc[i] == -2:  # Sell signal
        entry_price = data['Close'].iloc[i]
        for j in range(i + 1, len(data)):
            price_change = (entry_price - data['Close'].iloc[j]) / entry_price
            if price_change >= min_profit or price_change <= -min_profit:
                data['Profit'].iloc[i] = price_change
                break

# Label signals as profitable (1) or not (0)
data['Profitable'] = np.where(data['Profit'] >= min_profit, 1, -1)


# Function to compute rolling mean for the last 5 non-NaN values
def rolling_mean_last_5_non_na(series):
    non_na_series = series.dropna()
    if len(non_na_series) == 0:
        return None
    return non_na_series.tail(5).mean()


# Apply a rolling window and use the custom function
data['Last10CandlePerf'] = data['Close'].pct_change().rolling(window=10).mean()

# Identify bullish candles
data['Bullish'] = (data['Close'] > data['Open']).astype(int)

# Identify bearish candles
data['Bearish'] = (data['Close'] < data['Open']).astype(int)

# Calculate rolling sum of bullish candles over the last 20 periods
data['Bullish_Sum'] = data['Bullish'].rolling(window=20).sum()

# Calculate rolling sum of bearish candles over the last 20 periods
data['Bearish_Sum'] = data['Bearish'].rolling(window=20).sum()

# Calculate BullishRatio
data['BullishRatio'] = data['Bullish_Sum'] / (data['Bullish_Sum'] + data['Bearish_Sum'])

# Handle division by zero by replacing NaN with 0
data['BullishRatio'] = data['BullishRatio'].fillna(0)


# Prepare the dataset for the Bayesian Network
df = pd.DataFrame(data=data, columns=['hour', 'BullishRatio','Profitable','Last10CandlePerf'])
df = df.fillna(0)

# Discretize the continuous data (Hour) into 24 bins (0=> 00:00, 13=> 13:00)
discretizer_hour = KBinsDiscretizer(n_bins=24, encode='ordinal', strategy='uniform')
df['Hour_Binned'] = discretizer_hour.fit_transform(df[['hour']])

discretizer_bullishratio = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
df['BullishRatio_Binned'] = discretizer_bullishratio.fit_transform(df[['BullishRatio']])

discretizer_last10ref = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
df['Last10CandlePerf_Binned'] = discretizer_last10ref.fit_transform(df[['Last10CandlePerf']])

# Drop the original continuous 'hour','BullishRatio','Last10CandlePerf' column since it's been discretized
df = df.drop(columns=['hour','BullishRatio','Last10CandlePerf'])
data2 = df

# Learn the structure using the Hill Climbing algorithm
hc = HillClimbSearch(data2)
best_model = hc.estimate(scoring_method=BicScore(data2))

# Print the learned structure
print("Learned Bayesian Network Structure:")
print(best_model.edges())

# Fit the model with Conditional Probability Distributions (CPDs)
model = BayesianNetwork(best_model.edges())
model.fit(data2, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

# Define the learned structure as a list of edges (directed connections)
edges = best_model.edges()

# Create a directed graph
G = nx.DiGraph()

# Add the edges to the graph
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(10, 8))  # Slightly larger figure for better visibility
pos = nx.spring_layout(G, seed=42)  # Add a seed for consistent positioning

# Define node colors based on the variable type
node_colors = ['lightgreen' if 'Binned' in node else 'lightblue' for node in G.nodes()]

plt.gca().set_facecolor((211/255, 202/255, 207/255, 0.8))

# Draw the graph with professional style adjustments
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=3500, font_size=10, font_weight='bold', arrowsize=25,
        edge_color='gray', width=2.5)  # Increased edge width for clarity

# Add edge labels (optional - currently empty labels)
edge_labels = {edge: "" for edge in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_color='red', font_size=9)

# Add a clear, professional plot title
plt.title("Bayesian Network Structure", fontsize=14, fontweight='bold', pad=20)

# Show the plot
plt.show()
