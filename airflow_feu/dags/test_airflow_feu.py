from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import biblio_test_jenkins

def hello_feu():
    print("🔥 Airflow Projet Feu est opérationnel !")
    # On tente d'écrire un petit fichier pour tester les permissions
    with open("/opt/airflow/test_ok.txt", "w") as f:
        f.write("Validation du DAG le " + str(datetime.now()))
    return "Succès"

with DAG(
    dag_id='test_simple_airflow_feu',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Lancement manuel uniquement
    catchup=False,
    tags=['debug']
) as dag:

    task_hello = PythonOperator(
        task_id='dire_hello',
        python_callable=hello_feu
    )