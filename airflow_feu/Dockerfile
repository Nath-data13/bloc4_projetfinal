

FROM apache/airflow:2.9.2-python3.10

USER airflow

# On garde tes versions qui fonctionnent
RUN pip install --no-cache-dir \
    pytest \
    xgboost==3.0.5 \
    scikit-learn \
    joblib \
    requests \
    mlflow==3.6.0 \
    pandas==2.3.0 \
    numpy==1.26.3 \
    folium \
    boto3 \
    python-dotenv

# On laisse le docker-compose gérer les volumes et les commandes
WORKDIR /opt/airflow