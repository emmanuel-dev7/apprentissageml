# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Titre et introduction ---
st.title("Prédiction du Prix des Logements")
st.write("""
Bienvenue sur l'application de prédiction des prix immobiliers.  
Saisissez les caractéristiques d'un logement pour obtenir une estimation de son prix.  
Cette application utilise un modèle de régression entraîné sur le dataset Kaggle House Prices.
""")

# Chemin absolu du dossier courant de app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le modèle
model_path = os.path.join(current_dir, "..", "models", "model.pkl")

# Charger le modèle
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Chemin absolu du dossier courant de app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le CSV
csv_path = os.path.join(current_dir, "..", "data", "raw", "train.csv")

# Charger le dataset
data = pd.read_csv(csv_path)

# --- Inputs pour la prédiction ---
st.subheader("Saisissez les caractéristiques du logement")

col1, col2 = st.columns(2)
with col1:
    grlivarea = st.number_input("Surface habitable (GrLivArea)", min_value=0)
    garagecars = st.number_input("Nombre de voitures dans le garage (GarageCars)", min_value=0)
    totalbsmt = st.number_input("Surface du sous-sol (TotalBsmtSF)", min_value=0)
with col2:
    overallqual = st.slider("Qualité générale (OverallQual)", min_value=1, max_value=10)
    yearbuilt = st.number_input("Année de construction (YearBuilt)", min_value=1800, max_value=2025)
    totallot = st.number_input("Surface du terrain (LotArea)", min_value=0)

# --- Bouton de prédiction ---
if st.button("Prédire le prix"):
    input_data = pd.DataFrame({
        "GrLivArea": [grlivarea],
        "OverallQual": [overallqual],
        "GarageCars": [garagecars],
        "TotalBsmtSF": [totalbsmt],
        "YearBuilt": [yearbuilt],
        "LotArea": [totallot]
    })
    prediction = model.predict(input_data)
    st.success(f"Le prix estimé du logement est : {prediction[0]:,.2f} €")

# --- Graphique 1 : Distribution des prix ---
st.subheader("Distribution des prix dans le dataset")
fig, ax = plt.subplots()
sns.histplot(data['SalePrice'], kde=True, bins=30, ax=ax)
ax.set_xlabel("Prix")
ax.set_ylabel("Nombre de logements")
st.pyplot(fig)

# --- Graphique 2 : Importance des variables (si modèle arbre) ---
if hasattr(model, "feature_importances_"):
    st.subheader("Importance des variables")
    features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"]
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax2)
    st.pyplot(fig2)

# --- Graphique 3 : Prédiction vs valeurs réelles ---
st.subheader("Prédictions vs valeurs réelles sur le dataset")
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"]
X = data[features]
y = data["SalePrice"]
y_pred = model.predict(X)
fig3, ax3 = plt.subplots()
ax3.scatter(y, y_pred, alpha=0.5)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax3.set_xlabel("Valeurs réelles")
ax3.set_ylabel("Prédictions")
st.pyplot(fig3)
