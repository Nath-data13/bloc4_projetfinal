import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
from xgboost import XGBRegressor
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


# config AWS et chargement données 
load_dotenv("../secrets.env")

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

# Lecture directe depuis S3
obj = s3.get_object(Bucket=BUCKET_NAME, Key="dataset_complet_meteo.csv")
csv_str = obj['Body'].read().decode('utf-8')  # convertir les bytes en str
# df = pd.read_csv(StringIO(csv_str), sep=";")
# # Ajoute low_memory=False pour le CSV
df = pd.read_csv(StringIO(csv_str), sep=";", low_memory=False)

# config mlflow
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("projet_feu")

# modéle xgboost survival
def train_model(df):
    # df = load_model_data()
    df = df.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    df["event"] = df["event"].astype(bool)
    df["duration"] = df["duration"].fillna(0)

    features = [
        "moyenne precipitations mois", "moyenne temperature mois",
        "moyenne evapotranspiration mois", "moyenne vitesse vent année",
        "moyenne vitesse vent mois", "moyenne temperature année",
        "RR", "UM", "ETPMON", "TN", "TX", "Nombre de feu par an",
        "Nombre de feu par mois", "jours_sans_pluie", "jours_TX_sup_30",
        "ETPGRILLE_7j", "compteur jours vers prochain feu",
        "compteur feu log", "Année", "Mois",
        "moyenne precipitations année", "moyenne evapotranspiration année",
    ]
    features = [f for f in features if f in df.columns]

    y_struct = Surv.from_dataframe("event", "duration", df)
    X_train, X_test, y_train, y_test = train_test_split(df[features], y_struct, test_size=0.3, random_state=42)
    ev_train, du_train = y_train["event"], y_train["duration"]
    ev_test, du_test = y_test["event"], y_test["duration"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            objective="survival:cox",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            tree_method="hist",
            random_state=42,
        )),
    ])

    with mlflow.start_run(run_name="XGBSurv_Train"):
        # 1. Entraînement
        model.fit(X_train, du_train, xgb__sample_weight=ev_train)

        # 2. Log hyperparamètres
        mlflow.log_params({"n_estimators":100, "learning_rate":0.05, "max_depth":3, "tree_method":"hist"})

        # 3. Calcul des prédictions et métriques
        log_hr_test = model.predict(X_test)
        c_index = concordance_index_censored(ev_test, du_test, log_hr_test)[0]
        print(f"C-index (test) : {c_index:.3f}")
        mlflow.log_metric("c_index_test", c_index)

        # # 4. Signature et Log du modèle (L'optimisation)
        # from mlflow.models.signature import infer_signature
        # signature = infer_signature(X_test, log_hr_test)
    
        # mlflow.sklearn.log_model(
        #     model, 
        #     artifact_path="survival_xgb_model",
        #     signature=signature
        # )

        # 4. Signature et Log du modèle avec enregistrement automatique
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_test, log_hr_test)
    
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="survival_xgb_model",
            signature=signature,
            # CETTE LIGNE ENREGISTRE LE MODÈLE DANS LE REGISTRY
            registered_model_name="survival_xgb_model" 
        )

        # 5. Création des fichiers visuels et CSV
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(log_hr_test, bins=30, ax=ax)
        ax.set_title("Distribution log hazard (test)")
        fig_path = "log_hazard_test.png"
        fig.savefig(fig_path)
        plt.close(fig)

        df_test_pred = pd.DataFrame({"duration": du_test, "event": ev_test, "log_hazard_pred": log_hr_test})
        csv_path = "predictions_test.csv"
        df_test_pred.to_csv(csv_path, index=False)

        # 6. Envoi des fichiers (Artifacts) vers HF/S3
        mlflow.log_artifact(fig_path)
        mlflow.log_artifact(csv_path)

        # 7. Nettoyage local (pour rester propre sur ton PC)
        os.remove(fig_path)
        os.remove(csv_path)

    print("Tout est sur Hugging Face et S3.")

if __name__ == "__main__":
    train_model(df)

# tester dans le terminal
#     cd model
#     python train.py