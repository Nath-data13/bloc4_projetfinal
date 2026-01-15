#-------------------------------------------------------- Imports nécessaires ---------------------------------------------------
import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import boto3
import mlflow
import mlflow.pyfunc

from dotenv import load_dotenv
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import set_config

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
set_config(display="text")


# #-------------------------------------------------------- Configuration AWS S3 ---------------------------------------------------
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = "eu-west-3"

s3 = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


#-------------------------------------------------------- Streamlit page config ---------------------------------------------------
st.set_page_config(page_title="Projet Incendies", layout="wide")

#-------------------------------------------------------- Sidebar navigation ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Accueil",
    "Notre Projet",
    "Résultats des modèles",
])

#-------------------------------------------------------- Footer ---------------------------------------------------
def show_footer():
    st.markdown("---")
    st.markdown("Projet réalisé dans le cadre de la formation Lead Data Scientist. © 2025")

#-------------------------------------------------------- Chargement des datasets depuis S3 ---------------------------------------------------
@st.cache_data(show_spinner="🔄 Téléchargement des dernières données météo…", ttl=None)

def load_model_data() -> pd.DataFrame:
    bucket = "myprojetfeu"
    key = "dataset_complet_meteo.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'], sep=';')
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du dataset modèle : {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="🔄 Téléchargement du dataset historique…", ttl=None)
def load_df_merge() -> pd.DataFrame:
    bucket = "myprojetfeu"
    key = "historique_incendies_avec_coordonnees.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'], sep=';', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du dataset historique : {e}")
        return pd.DataFrame()

#-------------------------------------------------------- MLflow ---------------------------------------------------
# Utilise ton secret HF pour l'URL
mlflow.set_tracking_uri("https://nath13huggingface-mlflow-feu-serveur.hf.space")


@st.cache_resource(show_spinner="🔄 Chargement du modèle prédictif…") # Utilise cache_resource pour les objets lourds
def load_mlflow_model():
    # 'latest' permet de toujours charger le dernier modèle entraîné sans changer l'ID
    model_uri = 'models:/survival_xgb_model/latest' 
    # Si tu n'as pas encore enregistré le modèle dans le Model Registry, 
    # utilise le chemin vers ton run spécifique :
    # model_uri = 'runs:/TON_RUN_ID_A_COPIER_DE_MLFLOW/survival_xgb_model'
    model = mlflow.pyfunc.load_model(model_uri)
    return model

_model = load_mlflow_model()

# mlflow.set_tracking_uri("https://djohell-ml-flow.hf.space")

# @st.cache_data(show_spinner="🔄 Chargement du modèle prédictif…", ttl=None)
# def load_mlflow_model():
#     model_uri = 'runs:/69a3c889954f4ce9a2139a4fb4cefc59/survival_xgb_model'
#     model = mlflow.pyfunc.load_model(model_uri)
#     return model

# _model = load_mlflow_model()
# st.write("✅ Modèle prédictif chargé avec nos dernières prévisions !")

#-------------------------------------------------------- Prédiction via MLflow ---------------------------------------------------
@st.cache_data(show_spinner="⚙️ Prédiction des risques…", ttl=None)
def predict_risk(df_raw: pd.DataFrame, _model) -> pd.DataFrame:
    df = df_raw.copy()
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

    log_hr_all = _model.predict(df[features])
    HR = np.exp(log_hr_all)

    def S0(t):
        return np.exp(-t/1000)

    horizons = {7:"proba_7j", 30:"proba_30j", 60:"proba_60j", 90:"proba_90j", 180:"proba_180j"}
    for t, col in horizons.items():
        df[col] = 1 - (S0(t) ** HR)

    for col in ["latitude","longitude","ville"]:
        if col not in df.columns:
            df[col] = np.nan

    df_map = df[["latitude","longitude","ville"] + list(horizons.values())].copy()
    return df_map

#-------------------------------------------------------- Page Accueil ---------------------------------------------------
if page == "Accueil":
    st.title("Carte du risque d’incendie en Corse")

    df_raw = load_model_data()
    df_map = predict_risk(df_raw, _model)

    horizons_lbl = {
        "7 jours":"proba_7j",
        "30 jours":"proba_30j",
        "60 jours":"proba_60j",
        "90 jours":"proba_90j",
        "180 jours":"proba_180j",
    }
    choix = st.radio("Choisissez l’horizon temporel souhaité :", list(horizons_lbl.keys()), horizontal=True, index=0)
    col_proba = horizons_lbl[choix]

    vmax = float(df_map[col_proba].max())
    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        hover_name="ville",
        hover_data={col_proba: ":.2%"},
        color=col_proba,
        color_continuous_scale="YlOrRd",
        range_color=(0.0, vmax),
        zoom=7,
        height=650,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(title="Probabilité", tickformat=".0%"),
    )
    st.subheader(f"Risque d’incendie – horizon **{choix}**")
    st.plotly_chart(fig, use_container_width=True)

    show_footer()

#-------------------------------------------------------- Page Notre Projet ---------------------------------------------------
elif page == "Notre Projet":
    st.title("🔥 Projet Analyse des Incendies 🔥")

    st.subheader(" 📊 Contexte")
    st.subheader("🌲La forêt française en chiffres")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
La France est le 4ᵉ pays européen en superficie forestière, avec **17,5 millions d’hectares** en métropole (32 % du territoire) et **8 millions** en Guyane.
Au total, les forêts couvrent environ **41 %** du territoire national.
- **75 %** des forêts sont privées (3,5 millions de propriétaires).
- **16 %** publiques (collectivités).
- **9 %** domaniales (État).
La forêt française est un réservoir de biodiversité :  
- **190 espèces d’arbres** (67 % feuillus, 33 % conifères).  
- **73 espèces de mammifères**, **120 d’oiseaux**.  
- Environ **30 000 espèces de champignons et autant d’insectes**.  
- **72 %** de la flore française se trouve en forêt.
Les forêts françaises absorbent environ **9 %** des émissions nationales de gaz à effet de serre, jouant un rôle crucial dans la lutte contre le changement climatique.
Le Code forestier encadre leur gestion durable pour protéger la biodiversité, l’air, l’eau et prévenir les risques naturels.
        """)

    st.header("🔥 Corse : Bilan Campagne Feux de Forêts 2024")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 Contexte","🛠️ Prévention","🚒 Moyens","📊 Statistiques","🔍 Causes","🔎 Enquêtes"])

    # Contenus des tabs
    with tab1:
        with st.expander("📌 Contexte général"):
            st.markdown("""
- **80 %** de la Corse est couverte de forêts/maquis → **fort risque incendie**  
- **2023-2024** : la plus chaude et la plus sèche jamais enregistrée  
- **714 mm** de pluie sur l’année (**78 %** de la normale)  
- **Façade orientale** : seulement **30 %** des précipitations normales
            """)

    with tab2:
        with st.expander("🛠️ Prévention & Investissements"):
            st.markdown("""
- **1,9 million €** investis en 2023-2024 par l’État (jusqu’à 80 % de financement)  
- Travaux financés :  
  - Pistes DFCI/DECI (Sorio di Tenda, Oletta, Île-Rousse…)  
  - Citernes souples & points d’eau  
  - Drones, caméras thermiques, logiciels SIG  
  - Véhicules pour réserves communales
            """)

    with tab3:
        with st.expander("🚒 Moyens déployés"):
            st.markdown("""
- Jusqu’à **500 personnels mobilisables**  
- **168 sapeurs-pompiers SIS2B**, **261 UIISC5**, forestiers-sapeurs, gendarmerie, ONF…  
- Moyens aériens :  
  - **1 hélico**, **2 canadairs** à Ajaccio  
  - **12 canadairs** + **8 Dashs** nationaux en renfort
            """)

    with tab4:
        with st.expander("📊 Statistiques Feux Été 2024"):
            st.markdown("""
- **107 feux** recensés (~9/semaine)  
- **130 ha** brûlés dont :  
  - 83 % des feux <1 ha : **5,42 ha**  
  - 4 gros feux >10 ha : **72,84 ha**  
  - Linguizetta (**22,19 ha**), Oletta (**18,9 ha**), Pioggiola (**18,75 ha**), Tallone (**13 ha**)  
- Depuis janvier 2024 : **285 feux** pour **587 ha**  
- Feu majeur à Barbaggio : **195 ha (33 % du total annuel)**
            """)

    with tab5:
        with st.expander("🔍 Causes des feux (38 cas identifiés)"):
            st.markdown("""
- **11** : foudre  
- **8** : écobuages  
- **6** : malveillance  
- **5** : accidents  
- **4** : mégots de cigarette
            """)
        with st.expander("⚠️ Prévention = priorité absolue"):
            st.markdown("""
- **90 %** des feux ont une origine humaine  
- Causes principales : **imprudences** (mégots, BBQ, travaux, écobuages…)
            """)

    with tab6:
        with st.expander("🔎 Enquêtes & Surveillance"):
            st.markdown("""
- **20 incendies** étudiés par la Cellule Technique d’Investigation (CTIFF)  
- Équipes mobilisées : **7 forestiers**, **15 pompiers**, **21 forces de l’ordre**  
- **Fermeture de massif** enclenchée 1 seule fois : forêt de Pinia
            """)

 #---------------------------------------------------Notre Objectif --------------------------------------------------------
  
    st.subheader("🎯 Notre Objectif")
    st.markdown("""
Dans un contexte de **changement climatique** et de **risques accrus d’incendies de forêt**, notre équipe a développé un projet innovant visant à **analyser et prédire les zones à risque d’incendie** en France, avec un focus particulier sur la **Corse**.
    """)
#---------------------------------------------------Obectifs du projet---------------------------------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔍 Exploration des données")
        st.markdown("""
- ✅ **Évolution du nombre d’incendies**, répartition par mois et par causes.
- ✅ **Cartographie interactive** des incendies sur tout le territoire.
- ✅ **Analyse des clusters** grâce à DBSCAN pour identifier les zones les plus à risque.
        """)

    with col2:
        st.subheader("📈 Modèles prédictifs")
        st.markdown("""
- ✅ **Comparaison des modèles** : Random Forest, XGBoost, analyse de survie.
- ✅ **Prédiction des zones à risque** avec visualisation sur carte.
- ✅ Fourniture d'un **outil décisionnel** pour les autorités et les services de gestion des risques.
        """)

    st.subheader("📘 Définition de l'analyse de survie (Survival Analysis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🧠 Qu’est-ce que l’analyse de survie ?")
        st.markdown("""
L’**analyse de survie** (ou **Survival Analysis**) est une méthode statistique utilisée pour **modéliser le temps avant qu’un événement se produise**, comme :
- 🔥 un incendie,
- 🏥 un décès,
- 📉 une résiliation d’abonnement,
- 🧯 une panne.
""")

    with col2:
        st.markdown("### 📌 Objectif :")
        st.markdown("""
> Estimer la **probabilité qu’un événement ne se soit pas encore produit** à un instant donné.
""")

    with col3:
        st.markdown("### 🔑 Concepts fondamentaux : ")
        st.markdown("""
- ⏳ **Temps de survie (`T`)** : temps écoulé jusqu’à l’événement.
- 🎯 **Événement** : le phénomène qu’on cherche à prédire (feu, panne, décès...).
- ❓ **Censure** : l’événement **n’a pas encore eu lieu** durant la période d’observation.
- 📉 **Fonction de survie `S(t)`** : probabilité de "survivre" après le temps `t`.
- ⚠️ **Fonction de risque `h(t)`** : probabilité que l’événement se produise **immédiatement après `t`**, sachant qu’il ne s’est pas encore produit.
""")
    
    with col4:
        st.markdown ("### 🧪 Exemples d’applications :")
        st.markdown("""
| Domaine | Exemple |
|--------|---------|
| 🔥 Incendies | Quand un feu va-t-il se déclarer ? |
| 🏥 Santé | Combien de temps un patient survivra après traitement ? |
| 📉 Marketing | Quand un client risque-t-il de partir ? |
| 🧑‍💼 RH | Quand un salarié quittera-t-il l’entreprise ? |
""")

    #---------------------------------------------------Equipe du projet---------------------------------------------------
    st.subheader("👨‍💻 Équipe du projet")

    # Organisation par ordre alphabétique
    equipe = [
        "David Jaoui",
        "Francois Minaret",
        "Gilles Akakpo",
        "Joël Termondjian",
        "Marc Barthes",
        "Nathalie Devogelaere"
    ]

# Création de 3 colonnes pour une disposition aérée sur 2 lignes
    col1, col2, col3 = st.columns(3)

    for i, membre in enumerate(equipe):
        if i % 3 == 0:
            col1.markdown(f"✅ **{membre}**")
        elif i % 3 == 1:
            col2.markdown(f"✅ **{membre}**")
        else:
            col3.markdown(f"✅ **{membre}**")

    show_footer()


#---------------------------------------------------- Page Résultats des modèles -----------------------------------------
elif page == "Résultats des modèles":
    st.title("📈 Résultats des modèles prédictifs")
    st.markdown("### Comparaison des modèles de Survival Analysis")

    #--------------------------------------------------- Tableau codé en dur en Markdown -----------------------------------
    st.markdown("""
    | Modèle                            | Concordance Index | 
    |-----------------------------------|-------------------|
    | Predict survival fonction (MVP)   | 0.69              |                 
    | XGBOOST survival cox              | 0.809             |      
    """)

    st.markdown("👉 Le modèle **XGBOOST survival cox** obtient la meilleure performance globale.")

    show_footer()
