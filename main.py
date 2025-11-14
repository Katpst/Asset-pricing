import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_caps_quotes.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

k_values = [0.0, 0.02, 0.03, 0.05]
df = df[df['k'].isin(k_values)]

plt.figure(figsize=(10,6))
for k in k_values:
    subset = df[df['k'] == k]
    plt.plot(subset['date'], subset['price_per_1'], marker='o', label=f'k={k}')

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



