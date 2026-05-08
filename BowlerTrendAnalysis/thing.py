# %% [markdown]
# # Predicting the Bowling Order in IPL Using Multiple Regression
# 
# A beginner-friendly pipeline that uses Multiple Linear Regression to predict a bowler's 
# "average over" based on multiple stats, and then smartly allocates a 20-over innings.

# %% [markdown]
# ## 1. Load Data and Calculate Global Stats
# We load the dataset and compute multiple features for each bowler: Economy Rate, Wickets, and Total Balls Bowled.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load deliveries data
df = pd.read_csv("deliveries.csv")
df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

# Calculate multiple stats for all bowlers (Feature Engineering)
bowler_stats = df.groupby("bowler").agg(
    runs_conceded=("total_runs", "sum"),
    wickets_taken=("is_wicket", "sum"),
    balls_bowled=("ball", "count")
).reset_index()

# Convert balls to overs and calculate economy rate
bowler_stats["overs_bowled"] = bowler_stats["balls_bowled"] / 6
bowler_stats["economy_rate"] = bowler_stats["runs_conceded"] / bowler_stats["overs_bowled"]

bowler_stats.head()

# %% [markdown]
# ## 2. Isolate a Team & Define the Target Variable
# We will focus on Mumbai Indians. Our target variable ($y$) for the regression is the historical "average over" a bowler bowls.

# %%
team_name = "Mumbai Indians"
team_df = df[df["bowling_team"] == team_name].copy()

# Calculate the target variable: the average over bowled by each player for this team
avg_overs = team_df.groupby("bowler")["over"].mean().reset_index()
avg_overs.rename(columns={"over": "historical_avg_over"}, inplace=True)

# Merge our global stats with the team target variable
model_df = bowler_stats.merge(avg_overs, on="bowler")

# Filter for frontline bowlers (e.g., those who have bowled at least 10 overs / 60 balls)
model_df = model_df[model_df["balls_bowled"] >= 60].copy()

print(f"Total regular bowlers found for {team_name}: {len(model_df)}")
model_df.head()

# %% [markdown]
# ## 3. Train the Multiple Regression Model
# We train the model using three features to predict the phase of the innings they usually bowl in.

# %%
# X = Multiple Features (Economy, Wickets, Workload)
# y = Target (Historical Average Over)
X = model_df[["economy_rate", "wickets_taken", "balls_bowled"]]
y = model_df["historical_avg_over"]

# Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict the average over for these bowlers based on the model
model_df["predicted_avg_over"] = model.predict(X)

# Sort bowlers by their predicted phase (lowest = powerplay, highest = death overs)
model_df = model_df.sort_values("predicted_avg_over")

# Select the top 5 bowlers to form our core bowling attack
top_5_bowlers = model_df.head(5)["bowler"].tolist()
print("Selected Top 5 Bowlers for the innings:", top_5_bowlers)

# %% [markdown]
# ## 4. The Smart Prediction Pipeline
# We allocate 20 overs. For each over, we find the bowler whose `predicted_avg_over` is closest to the current over, enforcing two rules:
# 1. Max 4 overs per bowler.
# 2. No consecutive overs.

# %%
overs_schedule = []
bowler_overs_count = {b: 0 for b in top_5_bowlers}

for current_over in range(20):
    best_bowler = None
    min_diff = float('inf')
    
    for bowler in top_5_bowlers:
        # Rule 1: Max 4 overs per bowler
        if bowler_overs_count[bowler] < 4:
            
            # Rule 2: Cannot bowl consecutive overs
            if current_over == 0 or overs_schedule[-1] != bowler:
                
                # Find the bowler whose predicted average is closest to this current match over
                pred_over = model_df[model_df["bowler"] == bowler]["predicted_avg_over"].iloc[0]
                diff = abs(pred_over - current_over)
                
                if diff < min_diff:
                    min_diff = diff
                    best_bowler = bowler
                    
    overs_schedule.append(best_bowler)
    bowler_overs_count[best_bowler] += 1

# Create a clean DataFrame of the predicted innings
predicted_innings = pd.DataFrame({
    "Over": range(1, 21), # 1 to 20 for readability
    "Predicted_Bowler": overs_schedule
})

predicted_innings

# %% [markdown]
# ## 5. Visualization
# Plot the over-by-over timeline to visualize the spells.

# %%
plt.figure(figsize=(10, 5))

# Assign distinct colors to our 5 bowlers
colors = sns.color_palette("Set1", len(top_5_bowlers))
color_map = dict(zip(top_5_bowlers, colors))

# Plot the timeline
for i, row in predicted_innings.iterrows():
    plt.scatter(row['Over'], row['Predicted_Bowler'], 
                color=color_map[row['Predicted_Bowler']], s=200)

plt.plot(predicted_innings['Over'], predicted_innings['Predicted_Bowler'], 
         linestyle='--', alpha=0.4, color='gray')

plt.title(f"Predicted Bowling Timeline ({team_name})")
plt.xlabel("Over Number")
plt.ylabel("Bowler")
plt.xticks(range(1, 21))
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()