"""Training + visualisation pour le concours House Prices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def train_and_save(root: str | Path = None, output_path: str | Path = None):
    from ..data.make_dataset import load_raw_data
    from ..features.build_features import prepare_features

    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    base = Path(root) if root else Path.cwd()
    print("Chargement des données...")
    train_df, test_df = load_raw_data(base)

    print("Préprocessing...")
    X_train_t, y_train, X_test_t, preprocessor, test_ids = prepare_features(train_df, test_df)

    print("Entraînement du RandomForest...")
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)

    # === PRÉDICTIONS SUR LE TRAIN POUR VISUALISATION ===
    y_pred_train = model.predict(X_train_t)

    # === MÉTRIQUES ===
    rmse = root_mean_squared_error(y_train, y_pred_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    print(f"\nRMSE sur train : {rmse:,.0f} $")
    print(f"MAE  sur train : {mae:,.0f} $")

    # === GRAPHIQUE 1 : Prédiction vs Réalité ===
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.6, s=20)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel("Prix réel (SalePrice)")
    plt.ylabel("Prix prédit")
    plt.title(f"Prédiction vs Réalité\nRMSE = {rmse:,.0f} $")
    plt.tight_layout()
    plt.savefig(base / "reports" / "figures" / "pred_vs_real.png", dpi=200)
    plt.show()

    # === GRAPHIQUE 2 : Top 20 Feature Importance ===
    # Récupérer les noms des colonnes après OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20

    plt.figure(figsize=(10, 10))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top 20 des variables les plus importantes (Random Forest)")
    plt.tight_layout()
    plt.savefig(base / "reports" / "figures" / "feature_importance.png", dpi=200)
    plt.show()

    # === SAUVEGARDE ===
    wrapper = {
        "preprocessor": preprocessor,
        "model": model,
        "test_ids": test_ids,
        "rmse_train": rmse,
    }

    out = Path(output_path) if output_path else base / "models" / "model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    (base / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    joblib.dump(wrapper, out)
    print(f"\nModèle + graphiques sauvegardés !")
    print(f"→ Modèle : {out}")
    print(f"→ Graphiques : reports/figures/")

    return out


if __name__ == "__main__":
    print("=== Entraînement + Visualisation House Prices ===\n")
    train_and_save()