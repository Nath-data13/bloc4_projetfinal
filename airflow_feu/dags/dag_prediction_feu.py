from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import pandas as pd
import os
import psycopg2

# Configuration
MODEL_NAME = "survival_xgb_model"
MODEL_VERSION = "1"
# On récupère les variables d'environnement validées par Jenkins
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI")

def run_prediction():
    # 1. Connexion au serveur MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    
    print(f"Chargement du modèle : {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # 2. Chargement des données (simulées ou depuis ton CSV)
    # Pour le test, on prend une ligne d'exemple
    data_test = pd.DataFrame([[0.5, 25.0, 15.0]], columns=['feature1', 'feature2', 'feature3'])
    
    # 3. Prédiction
    prediction = model.predict(data_test)
    probabilite = float(prediction[0])
    print(f"🔥 Risque d'incendie prédit : {probabilite}")

    # 4. Sauvegarde dans Neon DB
    try:
        conn = psycopg2.connect(BACKEND_STORE_URI)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (date, score) VALUES (%s, %s)",
            (datetime.now(), probabilite)
        )
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Résultat enregistré dans Neon DB")
    except Exception as e:
        print(f"⚠️ Erreur DB (optionnelle pour ce test) : {e}")

with DAG(
    dag_id='prediction_risque_incendie_corse',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlflow', 'xgboost', 'prediction']
) as dag:

    predict_task = PythonOperator(
        task_id='inference_task',
        python_callable=run_prediction
    )