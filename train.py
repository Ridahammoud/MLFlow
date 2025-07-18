# train.py

import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import numpy as np

from model import model

# Nettoyage pour entrainement sain du modèl LightGbm
def clean_data(df):
    # Convertir les colonnes numériques encodées en texte
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                continue  # laisse les colonnes catégorielles
    return df

def convertir_types(df):
    df = df.copy()
    
    # Convertir les chaînes vides en NaN
    df.replace('', np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # On tente une conversion vers float
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                pass  # Si ça échoue, on garde la colonne catégorielle
    return df

# Chargement des données
app_train = pd.read_csv("app_train.csv")
app_test = pd.read_csv("app_test.csv")

app_train = clean_data(app_train)
app_test = clean_data(app_test)

# 🏁 Lancement de l’entraînement + logs MLflow
submission, fi, metrics, feature_names, features, labels = model(app_train, app_test)

print(" Baseline metrics :")
print(metrics)

# Nettoyage des noms de colonnes pour éviter erreur JSON avec LightGBM
clean_feature_names = [name.replace('"', '_').replace("'", "_").replace(":", "_").replace(",", "_") for name in feature_names]

# Réentraînement final pour enregistrement dans le Model Registry
final_model = lgb.LGBMClassifier(
    n_estimators=10000,
    objective='binary',
    class_weight='balanced',
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    n_jobs=-1,
    random_state=50
)

features_df = pd.DataFrame(features, columns=clean_feature_names)
features_df = convertir_types(features_df)

final_model.fit(features_df, labels)

# Enregistrement du modèle final dans le Model Registry
mlflow.set_experiment("LightGBM_HomeCredit")

with mlflow.start_run(run_name="Final_Model") as run:
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name="LightGBM_HomeCredit_Model"
    )
    print(f" Modèle enregistré sous : LightGBM_HomeCredit_Model (run ID: {run.info.run_id})")
