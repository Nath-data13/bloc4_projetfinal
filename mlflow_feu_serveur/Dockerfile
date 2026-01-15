# FROM python:3.13.5-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt ./
# COPY src/ ./src/

# RUN pip3 install -r requirements.txt

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

# RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# le lien entre MLflow ↔ S3.
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
RUN unzip awscliv2.zip
RUN ./aws/install


COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${ARTIFACT_STORE_URI}" \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
    --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}"