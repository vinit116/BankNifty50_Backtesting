import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('TradeReport.csv')

Equity = data['Total_Capital_Return']
Time = data['Date']
Drawdown = data['Pfolio_Total_Daily_Drawdown']

plt.figure(figsize=(10, 6))

plt.plot(Time, Equity, color='b', linestyle='-')

plt.title('Equity Curve')  
plt.xlabel('Date')  
plt.ylabel('Equity (in lacs)')


tick_labels = Time.iloc[::10].tolist()
last_tick = Time.iloc[-1]
tick_labels.append(last_tick)

plt.xticks(rotation=90)  
plt.xticks(tick_labels)

plt.tight_layout() 

plt.show() 

plt.figure(figsize=(10, 6))

plt.plot(Time, Drawdown, color='r', linestyle='-')

plt.title('Drawdown Over Time')  
plt.xlabel('Date')  
plt.ylabel('Drawdown')

plt.xticks(rotation=90)  
plt.xticks(tick_labels)

plt.tight_layout() 

plt.show() 