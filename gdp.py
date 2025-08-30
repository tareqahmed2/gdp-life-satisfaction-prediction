import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli['Indicator']=='Life satisfaction']
    oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')
    gdp_per_capita.rename(columns={'2015':'GDP per capita'}, inplace=True)
    gdp_per_capita.set_index('Country', inplace=True)
    return pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)

oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',')  

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("GDP vs Life satisfaction")
plt.show()

model = LinearRegression()
model.fit(X, y)

X_new = [[22587]] 
predicted = model.predict(X_new)
print(f"Predicted Life Satisfaction for Cyprus: {predicted[0][0]:.2f}")
