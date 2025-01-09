import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# data set ko read krwa 
athlete_data = pd.read_csv('athlete_events.csv')
noc_data = pd.read_csv('noc_regions.csv')

#  NOC ke naam pr datasets ko merge kr le
merged_data = pd.merge(athlete_data, noc_data, on='NOC', how='left')

# humko sirf filter daal hai medal ka region or year ke basis pr
medal_data = merged_data[merged_data['Medal'].notna()]  #isme sirf rowa aayengi medal 
medal_data = medal_data.groupby(['Year', 'region'])['Medal'].count().reset_index()
medal_data.rename(columns={'region': 'Country', 'Medal': 'Total_Medals'}, inplace=True)

# Prediction ke liy model bnane ke liy code hai ye
pivot_data = medal_data.pivot(index='Year', columns='Country', values='Total_Medals').fillna(0)
pivot_data = pivot_data.reset_index()

# prediction years aane vale
future_years = [2020, 2024]
predictions_2020 = {}
predictions_2024 = {}

# model every country ke liy 
for country in pivot_data.columns[1:]:
    country_data = pivot_data[['Year', country]].dropna()
    X = country_data['Year'].values.reshape(-1, 1)
    y = country_data[country].values
    
   
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict medals for 2020 and 2024
    pred_2020 = model.predict(np.array([2020]).reshape(-1, 1))[0]
    pred_2024 = model.predict(np.array([2024]).reshape(-1, 1))[0]
    
    # non negative ke liy
    predictions_2020[country] = max(0, pred_2020)  
    predictions_2024[country] = max(0, pred_2024)

# top 5 countries for 2020 and 2024
top_5_2020 = sorted(predictions_2020.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_2024 = sorted(predictions_2024.items(), key=lambda x: x[1], reverse=True)[:5]

# results ko plot krwana hai 
def plot_top_5(top_5, year):
    countries, medals = zip(*top_5)
    plt.bar(countries, medals, color='gold')
    plt.title(f'Top 5 Predicted Countries for {year} Olympics')
    plt.xlabel('Country')
    plt.ylabel('Predicted Total Medals')
    plt.show()

# 2020 ke liy
plot_top_5(top_5_2020, 2020)

# 2024 ke liy
plot_top_5(top_5_2024, 2024)

#  results normally terminal me print krwane ke liy visulaizaton ke baad 
print("Top 5 Predicted Countries for 2020 Olympics:")
for i, (country, total) in enumerate(top_5_2020, 1):
    print(f"{i}. {country}: {total:.2f} medals")

print("\nTop 5 Predicted Countries for 2024 Olympics:")
for i, (country, total) in enumerate(top_5_2024, 1):
    print(f"{i}. {country}: {total:.2f} medals")