# """Training + visualisation  House Prices."""

# from __future__ import annotations

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from pathlib import Path
# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error


# def train_and_save(root: str | Path = None, output_path: str | Path = None):
#     from ..data.make_dataset import load_raw_data
#     from ..features.build_features import prepare_features

#     plt.style.use("seaborn-v0_8")
#     sns.set_palette("husl")

#     base = Path(root) if root else Path.cwd()
#     print("Chargement des données...")
#     train_df, test_df = load_raw_data(base)

#     print("Préprocessing...")
#     X_train_t, y_train, X_test_t, preprocessor, test_ids = prepare_features(train_df, test_df)

#     print("Entraînement du RandomForest...")
#     model = RandomForestRegressor(
#         n_estimators=400,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         random_state=42,
#         n_jobs=-1,
#     )
#     model.fit(X_train_t, y_train)

#     # === PRÉDICTIONS SUR LE TRAIN POUR VISUALISATION ===
#     y_pred_train = model.predict(X_train_t)

#     # === MÉTRIQUES ===
#     rmse = root_mean_squared_error(y_train, y_pred_train)
#     mae = mean_absolute_error(y_train, y_pred_train)
#     print(f"\nRMSE sur train : {rmse:,.0f} $")
#     print(f"MAE  sur train : {mae:,.0f} $")

#     # === GRAPHIQUE 1 : Prédiction vs Réalité ===
    
#     plt.figure(figsize=(10, 8))
#     plt.scatter(y_train, y_pred_train, alpha=0.6, s=20)
#     plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
#     plt.xlabel("Prix réel (SalePrice)")
#     plt.ylabel("Prix prédit")
#     plt.title(f"Prédiction vs Réalité\nRMSE = {rmse:,.0f} $")
#     plt.tight_layout()
#     plt.savefig(base / "reports" / "figures" / "pred_vs_real.png", dpi=200)
#     plt.show()

#     # === GRAPHIQUE 2 : Top 20 Feature Importance ===
#     # Récupérer les noms des colonnes après OneHotEncoding
#     feature_names = preprocessor.get_feature_names_out()

#     importances = model.feature_importances_
#     indices = np.argsort(importances)[-20:]  # Top 20

#     plt.figure(figsize=(10, 10))
#     plt.barh(range(len(indices)), importances[indices], align="center")
#     plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#     plt.xlabel("Importance")
#     plt.title("Top 20 des variables les plus importantes (Random Forest)")
#     plt.tight_layout()
#     plt.savefig(base / "reports" / "figures" / "feature_importance.png", dpi=200)
#     plt.show()

#     # === SAUVEGARDE ===
#     wrapper = {
#         "preprocessor": preprocessor,
#         "model": model,
#         "test_ids": test_ids,
#         "rmse_train": rmse,
#     }

#     out = Path(output_path) if output_path else base / "models" / "model.joblib"
#     out.parent.mkdir(parents=True, exist_ok=True)
#     (base / "reports" / "figures").mkdir(parents=True, exist_ok=True)

#     joblib.dump(wrapper, out)
#     print(f"\nModèle + graphiques sauvegardés !")
#     print(f"→ Modèle : {out}")
#     print(f"→ Graphiques : reports/figures/")

#     return out


# if __name__ == "__main__":
#     print("=== Entraînement + Visualisation House Prices ===\n")
#     train_and_save()
"""Training + visualisation  House Prices (Pipeline + Multi-model)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.inspection import permutation_importance

# Modèles avancés
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def train_and_save(root: str | Path = None, output_path: str | Path = None):
    from ..data.make_dataset import load_raw_data
    from ..features.build_features import prepare_features

    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    base = Path(root) if root else Path.cwd()
    print("Chargement des données...")
    train_df, test_df = load_raw_data(base)

    print("Préprocessing...")
    X_train, y_train, X_test, preprocessor, test_ids = prepare_features(train_df, test_df)

    # =============================
    # Définition des modèles
    # =============================

    MODELS = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    #  Ajout du stacking
    MODELS["Stacking"] = StackingRegressor(
        estimators=[
            ("rf", MODELS["RandomForest"]),
            ("xgb", MODELS["XGBoost"]),
            ("lgbm", MODELS["LightGBM"]),
        ],
        final_estimator=RandomForestRegressor(n_estimators=300, random_state=42),
        n_jobs=-1,
    )

    results = {}
    models_wrapped = {}

    # ========================================
    # Boucle d’entraînement pour chaque modèle
    # ========================================
    for name, model in MODELS.items():
        print(f"\n=== Entraînement du modèle : {name} ===")

        # Pipeline complet
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)

        # Prédictions
        y_pred = pipe.predict(X_train)

        rmse = root_mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)

        results[name] = {"rmse": rmse, "mae": mae}
        models_wrapped[name] = pipe

        print(f"► RMSE : {rmse:,.0f} $")
        print(f"► MAE  : {mae:,.0f} $")

        # ============================
        # Graphique : Prédiction vs Réalité
        # ============================
        plt.figure(figsize=(8, 8))
        plt.scatter(y_train, y_pred, alpha=0.6)
        plt.plot([y_train.min(), y_train.max()],
                 [y_train.min(), y_train.max()], "r--")
        plt.xlabel("Prix réel")
        plt.ylabel("Prix prédit")
        plt.title(f"{name} – Prédiction vs réalité (RMSE={rmse:,.0f}$)")
        plt.tight_layout()
        save_fig = base / "reports" / "figures" / f"{name}_pred_vs_real.png"
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=200)
        plt.close()

        # ============================
        # Importance des features (permutation)
        # ============================
        print("Calcul des importances (permutation)...")

        perm = permutation_importance(
            pipe,
            X_train,
            y_train,
            n_repeats=15,
            random_state=42,
            n_jobs=-1,
        )

        # Top 20
        idx = np.argsort(perm.importances_mean)[-20:]
        feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(idx)), perm.importances_mean[idx])
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
        plt.title(f"{name} – Top 20 features importantes")
        plt.tight_layout()
        plt.savefig(base / "reports" / "figures" / f"{name}_feature_importance.png", dpi=200)
        plt.close()

    # =============================
    # Sauvegarde de tous les modèles
    # =============================
    out = Path(output_path) if output_path else base / "models" / "all_models.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "models": models_wrapped,
        "results": results,
        "test_ids": test_ids,
    }, out)

    print("\n======= FIN DE L’ENTRAÎNEMENT =======")
    print(f"Modèles sauvegardés dans : {out}")
    print("Graphiques dans : reports/figures/")
    print("Résultats :")
    for m, r in results.items():
        print(f" - {m}: RMSE={r['rmse']:,.0f}, MAE={r['mae']:,.0f}")

    return out


if __name__ == "__main__":
    print("=== Entraînement Pipeline + Multi-Modèles House Prices ===")
    train_and_save()
