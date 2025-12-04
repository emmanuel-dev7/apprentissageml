# prediction prix logements

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

prediction des prix de logements

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── app/
│   └── app.py         <- Application Streamlit
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         prediction_prix_logements and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── prediction_prix_logements   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes prediction_prix_logements a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train_model.py      <- Code to train models
    │   └── train_pkl.py        <- Code to train model.pkl
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Méthodologie du modèle House Prices

### 1. Chargement et séparation
- `train.csv` → features + cible (`SalePrice`)
- `test.csv` → seulement features

### 2. Préprocessing (robuste et sans fuite de données)
- Suppression de la colonne `Id`
- Variables numériques → imputation médiane + StandardScaler
- Variables catégorielles → imputation "missing" + OneHotEncoder (`handle_unknown='ignore'`)

Le preprocessor est fitté sur **train + test concaténés** → même traitement des catégories rares.

### 3. Modèle
RandomForestRegressor (400 arbres) :
- Très robuste aux outliers
- Gère naturellement les interactions
- Pas besoin de tuning compliqué pour un bon score de base

### 4. Évaluation
RMSE calculé sur le set d’entraînement 
Ce score reflète la qualité de généralisation du modèle.

### 5. Prédiction & soumission
Le script `predict.py` charge le modèle + preprocessor → génère `submission.csv` prêt pour Kaggle
Le script `train_pkl.py` charge `train.csv` prépare les donnees (preprocessing) puis entraine le modele apres génère le `model.pkl` puis `app.py` charge `model.pkl` pour faire les prédictions

### Création d'une application Streamlit 
L’application app/app.py permet de :
Entrer les caractéristiques d’un logement :
- GrLivArea
- OverallQual
- GarageCars
- TotalBsmtSF
- YearBuilt
- LotArea
Visualiser plusieurs graphiques :
- Distribution des prix réels (SalePrice) : 
    * Met en évidence la forme globalement asymétrique des prix (skewed distribution).
    * Permet d’identifier d’éventuels outliers.
    * Donne une première idée de la complexité de la prédiction.
    * Aide à justifier la normalisation ou une transformation log éventuelle.
    Ce graphique est fondamental pour comprendre la variable cible avant l’entraînement du modèle.
- Importance des variables (si modèle basé arbre) :
    * D’interpréter le comportement du modèle,
    * De valider les choix de features,
    * D’ajuster éventuellement le preprocessing.
    Cette visualisation rend le modèle moins “boîte noire” et améliore la transparence des prédictions.
- Prédiction vs valeurs réelles sur le dataset d’entraînement :
    * La précision globale du modèle,
    * La dispersion des erreurs,
    * Les zones où le modèle surestime ou sous-estime les prix.
    Ce graphique est rapide, visuel et très utile pour juger de la performance du modèle sans entrer dans des métriques avancées.
Ces trois visualisations permettent :
- Une analyse exploratoire (EDA) simple,
- Une compréhension intuitive du comportement du modèle,
- Une évaluation visuelle de la qualité des prédictions.
Elles renforcent la crédibilité de l'application et facilitent l’interprétation des résultats.
Obtenir une prédiction :
Le bouton "Prédire le prix" utilise `model.pkl` pour afficher le prix estimé.


### Résultats attendus
- Score Kaggle ≈ 0.145 – 0.155 → Top 40–50%
- Avec XGBoost/LightGBM + un peu de tuning → facilement Top 20%
![image de comparaison ](image.png)


