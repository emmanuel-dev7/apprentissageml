# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Charger les données
train = pd.read_csv("data/raw/train.csv")
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"]
X_train = train[features]
y_train = train["SalePrice"]

# 2. Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Sauvegarder le modèle
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle sauvegardé !")