import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("clientes.csv")

X = df[["idade", "renda", "gastos"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo = KMeans(n_clusters=3, random_state=42)
modelo.fit(X_scaled)

df["cluster"] = modelo.labels_

print(df.head())
