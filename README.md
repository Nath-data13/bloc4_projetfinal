# 🔥 Prédiction du Risque de Feu en Corse à partir de Données Météo

## 🧠 Objectif

Ce projet vise à **prédire le risque d'incendie (feu)** dans le temps pour chaque zone géographique de la **Corse**, en s’appuyant sur un **modèle de survie** basé sur des données **météorologiques** et des **données d’historique d’incendies**.

---

## 🗂️ Données utilisées

### 🔸 Données d’incendies (BDIFF)
- Source : [BDIFF - Base de Données des Incendies de Forêt en France](https://bdiff.agriculture.gouv.fr/)
- Période : 2006 à 2024
- Variables :
  - Date et lieu du feu
  - Localisation (commune, latitude, longitude)

### 🔸 Données météorologiques
- Source : [Météo-France](https://donneespubliques.meteofrance.fr/)
- Données quotidiennes par station météo en Corse
- Variables :
  - Température, humidité, vent, précipitations, etc.
  - Données synchronisées avec les dates et localisations des feux

---

## ⚙️ Modélisation

### 📌 Problématique
> Estimer la **probabilité qu’un feu se déclenche dans une zone donnée à un horizon t (7j, 30j, 60j...)**, en fonction des conditions météo récentes.

### 🔍 Modèle principal
- **XGBoost Regressor** avec l’objectif `survival:cox` (modèle de survie)
---

## 🗺️ Visualisation

### 📍 Carte interactive
- Affichage du risque de feu par zone sur une carte (Plotly ScatterMapbox)
- Possibilité de sélectionner l’horizon temporel (7j, 30j, etc.)

---

## 📊 Évaluation

- **C-index (test)** : ~0.80
- Permet de mesurer la capacité du modèle à bien classer les zones par risque relatif.

---

## 👤 Auteurs

- Faycal Belambri, Joel Termondjian, Marc Barthes
- Développé avec Python, Scikit-learn, XGBoost, Lifelines, Plotly

---