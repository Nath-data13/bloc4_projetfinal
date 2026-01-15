---
title: Mlflow Feu Serveur
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---
## 🌲 Serveur MLflow - Projet Prévention Incendies Corse

Ce Space héberge le serveur de tracking **MLflow** pour le projet de modélisation du risque d'incendie.

### Configuration Architecture :
* **Backend Store** : PostgreSQL (Neon.tech) pour les métriques.
* **Artifact Store** : AWS S3 pour les modèles et graphiques.
* **Modèle** : XGBoost Survival (Cox Model).

### Utilisation :
Pour logger vos expériences vers ce serveur, utilisez :
`mlflow.set_tracking_uri("https://nath13huggingface-mlflow-feu-serveur.hf.space")`
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
