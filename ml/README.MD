# 📊 Rapport d’Analyse — Wine Quality Dataset

## 🟦 Introduction

Le dataset **Wine Quality** est l’un des jeux de données les plus utilisés en data science pour étudier la prédiction de la qualité du vin à partir de ses caractéristiques physico-chimiques. Issu de recherches universitaires portugaises, il constitue un excellent support d’analyse exploratoire, de modélisation statistique et d’apprentissage automatique.

Ce compte rendu vise à répondre aux questions fondamentales **quoi, comment, quand, où, qui**, tout en situant ce dataset dans un contexte scientifique plus large. Nous présenterons également des études similaires menées dans la littérature, démontrant l’importance croissante de la modélisation des préférences gustatives et du contrôle qualité basé sur les données.

---

## 📌 1. Quoi ? (Objet de l’étude)

Le fichier contient deux datasets :

- **winequality-red.csv** : 1 599 échantillons de vin rouge  
- **winequality-white.csv** : 4 898 échantillons de vin blanc  

Ces données permettent de modéliser la qualité du vin notée par des experts, à partir de **11 mesures chimiques** telles que l’acidité, l’alcool, le pH, la densité, etc.

---

## 🔧 2. Comment ? (Méthodologie utilisée)

Les données reposent sur deux types de mesures :

### 🔬 2.1 Mesures physico-chimiques  
Chaque vin est analysé selon 11 variables standardisées :

- fixed acidity  
- volatile acidity  
- citric acid  
- residual sugar  
- chlorides  
- free sulfur dioxide  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  

### 👅 2.2 Mesures sensorielles  
Un panel d’experts attribue une note de **0 à 10**, formant la variable cible : `quality`.

Ces données permettent :

- la **régression**,  
- la **classification**,  
- les analyses statistiques (corrélations, distributions),  
- l’entraînement de modèles prédictifs (**SVM, Random Forest, XGBoost**, etc.).

---

## 📅 3. Quand ? (Période)

- Données publiées en **2009**
- Collectées entre **2004 et 2007**

---

## 📍 4. Où ? (Lieu)

Les vins proviennent de la région de **Vinho Verde** (Nord du Portugal).  
Les analyses ont été réalisées :

- dans des laboratoires œnologiques portugais,  
- en collaboration avec l’**Université de Minho**.

---

## 👥 5. Qui ? (Auteurs)

Les recherches ont été menées par :

- Paulo Cortez  
- António Cerdeira  
- Fernando Almeida  
- Telmo Matos  
- José Reis  

Publication originale :  
**Cortez et al. (2009)** – *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems.

---

## 📂 6. Structure des données

| Variable | Description |
|---------|-------------|
| fixed acidity | Acides non volatils |
| volatile acidity | Acide acétique |
| citric acid | Acidité citrique |
| residual sugar | Sucre résiduel |
| chlorides | Salinité |
| free sulfur dioxide | Conservateur libre |
| total sulfur dioxide | Conservateur total |
| density | Densité |
| pH | Acidité |
| sulphates | Additif |
| alcohol | Taux d’alcool |
| quality | Note sensorielle |

---

## 🔍 7. Questions explorables

- Quels paramètres influencent le plus la qualité ?
- Y a-t-il une différence entre vins rouges et blancs ?
- Peut-on prédire la qualité avec un modèle ML ?
- L’alcool améliore-t-il réellement la note ?
- Quel est l’impact du sucre, du pH ou des sulfites ?

---

## 📚 8. Études similaires

### ✔ Fernandes et al. (2015)  
Random Forest + réseaux de neurones pour prédire la qualité du vin.

### ✔ Zhang & Ma (2019)  
Comparaison de **SVM, KNN, Boosting**.

### ✔ Wine Quality Prediction (2020–2023)  
Approches diverses : PCA, Gradient Boosting, Deep Learning, clustering.

### ✔ Rapports OIV  
Utilisation d’analyses statistiques similaires pour le contrôle qualité.

---
```
# ======================================================
#   ANALYSE STATISTIQUE + ACP + VISUALISATIONS COMPLETES
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import shapiro

# ------------------------------------------------------
# 1) Charger la BDD
# ------------------------------------------------------
df = pd.read_csv("winequality-white.csv", sep=';')
print("\nAperçu des données :")
print(df.head())

# ------------------------------------------------------
# 2) ANALYSE STATISTIQUE
# ------------------------------------------------------

print("\n===== STATISTIQUES DESCRIPTIVES =====")
print(df.describe().T)

# Valeurs manquantes
print("\n===== VALEURS MANQUANTES =====")
print(df.isnull().sum())

# ------------------------------------------------------
# 3) MATRICE DE CORRÉLATION (GRAPHE)
# ------------------------------------------------------
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Heatmap de la matrice de corrélation")
plt.show()

# ------------------------------------------------------
# 4) DISTRIBUTION DES VARIABLES (HISTOGRAMMES)
# ------------------------------------------------------
df.hist(figsize=(14,10), bins=30)
plt.suptitle("Distribution des variables", fontsize=15)
plt.show()

# ------------------------------------------------------
# 5) BOXPLOTS (OUTLIERS)
# ------------------------------------------------------
plt.figure(figsize=(14,10))
for i, col in enumerate(df.columns):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 6) TEST DE NORMALITÉ SHAPIRO-WILK
# ------------------------------------------------------
print("\n===== TEST SHAPIRO (Normalité) =====")
for col in df.columns:
    sample = df[col].sample(500, random_state=0)  # max pour Shapiro
    stat, p = shapiro(sample)
    print(f"{col}: p-value={p:.4f} -> {'Non normal' if p < 0.05 else 'Normal'}")

# ------------------------------------------------------
# 7) PREPARATION DES DONNÉES POUR L’ACP
# ------------------------------------------------------
X = df.drop(columns=['quality'])
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------
# 8) ACP
# ------------------------------------------------------
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

print("\n===== VARIANCE EXPLIQUÉE =====")
for i, v in enumerate(explained_var, start=1):
    print(f"PC{i} : {v*100:.2f}%")

# -------------------------
# SCREE PLOT
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.title("Scree plot - Variance expliquée")
plt.xlabel("Composante principale")
plt.ylabel("Variance expliquée")
plt.grid(True)
plt.show()

# -------------------------
# VARIANCE CUMULÉE
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.title("Variance expliquée cumulée")
plt.xlabel("Composantes")
plt.ylabel("Variance cumulée")
plt.grid(True)
plt.show()

# -------------------------
# PROJECTION PC1 vs PC2
# -------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(label="quality")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection ACP : PC1 vs PC2")
plt.grid(True)
plt.show()

# -------------------------
# BIPLOT (loadings + individus)
# -------------------------
loadings = pca.components_.T
features = X.columns

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.4)

for i, var in enumerate(features):
    plt.arrow(0, 0, loadings[i,0]*3, loadings[i,1]*3, 
              color='red', head_width=0.05)
    plt.text(loadings[i,0]*3.2, loadings[i,1]*3.2, var, color='red')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot ACP")
plt.axhline(0); plt.axvline(0)
plt.grid(True)
plt.show()

# -------------------------
# HEATMAP DES LOADINGS
# -------------------------
load_df = pd.DataFrame(loadings, index=features,
                       columns=[f"PC{i}" for i in range(1,6)])

plt.figure(figsize=(10,6))
sns.heatmap(load_df, annot=True, cmap="viridis")
plt.title("Loadings des variables sur les composantes principales")
plt.show()
```
# 📈 9. Description de la BDD, Code Python, Analyses Statistiques & Interprétations

Cette section présente :

- ✔ une **description complète de la base de données**,  
- ✔ le **code Python** utilisé pour l’analyse,  
- ✔ des **analyses statistiques détaillées**,  
- ✔ une **interprétation approfondie** de chaque graphique.

---

## 🗄️ 9.1 Description de la base de données (BDD)

La base de données **Wine Quality** contient deux fichiers :

- **winequality-red.csv** : 1 599 observations de vins rouges  
- **winequality-white.csv** : 4 898 observations de vins blancs  

Chaque observation correspond à un vin mesuré selon **11 caractéristiques physico-chimiques** :

- fixed acidity  
- volatile acidity  
- citric acid  
- residual sugar  
- chlorides  
- free sulfur dioxide  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  

La variable cible est :

- **quality** : note sensorielle (0 à 10)

Les données sont numériques, homogènes et adaptées aux analyses statistiques, ACP, régression, classification et machine learning.

---

## 💻 9.2 Code Python utilisé

Voici le code Python de base pour charger, fusionner et explorer les données :

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des deux datasets
red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

# Ajout d’une colonne 'type' pour distinguer les deux vins
red["type"] = "red"
white["type"] = "white"

# Fusion
df = pd.concat([red, white], ignore_index=True)

# Affichage des premières lignes
print(df.head())

# Statistiques descriptives
print(df.describe())

# Distribution de la qualité
print(df["quality"].value_counts().sort_index())

## 🧠 Conclusion

Le dataset **Wine Quality** constitue un excellent cas d’étude pour explorer la relation entre données chimiques objectives et évaluations sensorielles subjectives. Grâce à sa structure claire et à son adoption massive, il est idéal pour :

- l’analyse exploratoire,  
- la modélisation prédictive,  
- l’apprentissage supervisé,  
- la comparaison de modèles.

Les relations entre composition chimique et qualité perçue sont complexes mais modélisables, faisant de ce dataset un **benchmark incontournable** en data science.

---

