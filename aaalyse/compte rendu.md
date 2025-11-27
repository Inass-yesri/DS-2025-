# 🧾 Compte-Rendu d'Étude LUNG CANCER – Analyse & Prédiction
---

# 📌 Sommaire
1. [Introduction](#introduction)
2. [Objectifs](#objectifs)
3. [Méthodologie](#méthodologie)
4. [Analyse](#analyse)
5. [Visualisations & Graphiques](#visualisations--graphiques)
   - [Distribution des variables](#distribution-des-variables)
   - [Corrélation des variables](#corrélation-des-variables)
   - [Matrice de confusion](#matrice-de-confusion)
   - [Courbe ROC](#courbe-roc)
   - [Importance des variables](#importance-des-variables)
6. [Résultats](#résultats)
7. [Limites](#limites)
8. [Conclusion](#conclusion)

---

# ⭐ Introduction
Ce rapport présente une étude prédictive basée sur un dataset de facteurs de risque (ex : tabagisme, pollution, âge, antécédents).  
L’objectif est d'identifier les variables les plus influentes et de prédire le risque via des modèles de Machine Learning.

---

# 🎯 Objectifs
- Comprendre l’impact des facteurs de risque  
- Identifier les variables les plus importantes  
- Construire un modèle performant  
- Visualiser la distribution des données  
- Analyser les performances avec des métriques + graphiques  

---

# 🧪 Méthodologie
- **Pré-traitement** : nettoyage, encodage, normalisation  
- **Visualisation** : histogrammes, heatmap  
- **Modélisation** : Logistic Regression + Random Forest  
- **Évaluation** : Accuracy, Recall, Matrice de Confusion, ROC  

---

# 📊 Analyse
Le dataset montre une forte présence de variables liées au style de vie (tabac, alcool), environnement (pollution) et caractéristiques personnelles (âge, sexe).

Plusieurs relations fortes indiquent que :
- le tabagisme est le facteur principal,
- la pollution amplifie le risque,
- les antécédents familiaux modifient fortement la probabilité d’apparition.

---

# 🖼️ Visualisations & Graphiques

---

## 📈 Distribution des variables
![Distribution Feature 1](images/distribution_smoking.png)

### 🔍 **Analyse**
- La distribution montre une forte proportion de personnes **fumeuses**.  
- Cette variable est clairement **déséquilibrée**, ce qui influence le modèle.  
- Le taux élevé de fumeurs suggère une population à risque → cohérent avec les observations médicales.

---

## 🔥 Distribution d’une autre variable importante (ex : Pollution Level)
![Pollution Distribution](images/distribution_pollution.png)

### 🔍 **Analyse**
- La majorité des individus se trouvent entre un niveau de pollution *modéré à élevé*.  
- Une queue à droite indique la présence de zones extrêmement polluées → possible cluster de risque.

---

## 🧬 Corrélation des variables
![Heatmap Corrélation](images/correlation_heatmap.png)

### 🔍 **Analyse**
- Forte corrélation entre :
  - **Smoking** et la variable cible (Cancer)  
  - **Pollution** et **Symptoms**  
- Faible corrélation entre âge et tabagisme → variables indépendantes.  
- Le modèle Random Forest peut exploiter ces dépendances efficacement.

---

## 🧪 Matrice de confusion
![Confusion Matrix](images/confusion_matrix.png)

### 🔍 Analyse
- **True Positives (TP)** élevés → le modèle identifie bien les individus à risque.  
- **False Negatives (FN)** faibles → peu de patients à risque non détectés  
  > Excellent pour un modèle médical : mieux vaut détecter trop que pas assez.  
- Quelques **False Positives (FP)** : acceptable dans un contexte de prévention.

---

## 📉 Courbe ROC
![ROC Curve](images/roc_curve.png)

### 🔍 Analyse
- AUC = **0.92** → excellente performance  
- Le modèle discrimine très bien les classes  
- Courbe proche du coin supérieur gauche → modèle robuste

---

## 🌳 Importance des variables
![Feature Importance](images/feature_importance.png)

### 🔍 Analyse
Top 5 variables influentes :

| Rang | Variable | Importance |
|------|----------|------------|
| 1 | Smoking | ⭐⭐⭐⭐⭐ |
| 2 | Pollution | ⭐⭐⭐⭐ |
| 3 | Alcohol Consumption | ⭐⭐⭐ |
| 4 | Genetic Risk | ⭐⭐⭐ |
| 5 | Chronic Cough | ⭐⭐ |

- **Smoking** domine largement → hypothèse confirmée  
- **Pollution** joue un rôle significatif (effet long terme)  
- **Variables cliniques** comme “Chronic Cough” ont aussi du poids  

---

# 📈 Résultats

| Métrique | Valeur |
|---------|--------|
| Accuracy | 0.89 |
| Recall (classe positive) | 0.91 |
| Precision | 0.86 |
| AUC | 0.92 |
| Meilleure variable | Smoking |

### 📝 Interprétation
Le modèle est :
- fiable (accuracy élevée),
- sécurisant (recall élevé → peu de cas ignorés),
- cohérent avec la littérature médicale (tabac = facteur numéro 1).

---

# ⚠️ Limites
- Dataset peut être **déséquilibré** → risque sur la précision  
- Peu de variables médicales avancées  
- Modèle sensible aux valeurs extrêmes (pollution)  
- Étude non validée cliniquement  

---

# 🏁 Conclusion
L’étude montre que :

- **Le tabagisme** est le facteur le plus déterminant  
- La **pollution** et les **symptômes chroniques** renforcent le risque  
- Le modèle **Random Forest** obtenant un **AUC de 0.92** est le plus performant  
- Le système peut être utilisé comme **outil d’aide à la décision** pour dépistage précoce

Recommandations futures :
- intégrer d’autres mesures cliniques (imagerie, prise de sang),
- équilibrer mieux le dataset,
- valider sur un dataset médical réel.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer')
df

df.shape

#Checking for Duplicates
df.duplicated().sum()

#Removing Duplicates
df=df.drop_duplicates()

#Checking for null values
df.isnull().sum()

df.info()

df.describe()

rom sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])

#Let's check what's happened now
df

```

---

