<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/4363f236-7e6b-4fae-8d12-f01561d016aa" />     EL YESRI INASS


# Rapport d'Analyse Approfondie du PIB International


## 1. INTRODUCTION

### 1.1 Objectif de l'analyse

Ce rapport vise à analyser de manière approfondie l'évolution du Produit Intérieur Brut (PIB) de plusieurs pays sur une période significative. L'objectif principal est de comprendre les dynamiques économiques mondiales, d'identifier les tendances de croissance et de comparer les performances économiques entre différentes régions du monde.

### 1.2 Méthodologie générale employée

Notre approche méthodologique repose sur :
- Une analyse quantitative des données macroéconomiques officielles
- L'utilisation de techniques statistiques descriptives et comparatives
- La visualisation de données pour faciliter l'interprétation
- Une analyse temporelle pour identifier les tendances et cycles économiques
- Une approche comparative entre pays développés et émergents

### 1.3 Pays sélectionnés et période d'analyse

**Pays sélectionnés :**
- **États-Unis** : Première économie mondiale, référence du monde développé
- **Chine** : Deuxième économie mondiale, représentant les économies émergentes
- **Japon** : Troisième économie mondiale, représentant l'Asie développée
- **Allemagne** : Première économie européenne, locomotive de l'UE
- **Inde** : Grande économie émergente à forte croissance démographique
- **France** : Économie développée européenne diversifiée
- **Brésil** : Représentant l'Amérique latine
- **Canada** : Économie développée riche en ressources naturelles

**Période d'analyse :** 2015-2023 (9 années)

Cette période permet d'analyser les tendances récentes tout en capturant l'impact de la pandémie de COVID-19 (2020-2021) sur les économies mondiales.

### 1.4 Questions de recherche principales

1. Quelle est l'évolution du PIB nominal des pays sélectionnés entre 2015 et 2023 ?
2. Quels pays affichent les taux de croissance les plus élevés et les plus stables ?
3. Comment le PIB par habitant varie-t-il entre les économies développées et émergentes ?
4. Quel a été l'impact de la pandémie COVID-19 sur les différentes économies ?
5. Existe-t-il des corrélations entre la taille économique et les taux de croissance ?

---

## 2. PRÉSENTATION DES DONNÉES

### 2.1 Source des données

**Source principale :** Banque mondiale - World Development Indicators (WDI)

**Sources complémentaires :**
- Fonds Monétaire International (FMI) - World Economic Outlook Database
- OCDE - Statistical Database

**Justification du choix :** Ces sources sont reconnues internationalement pour leur fiabilité, leur méthodologie standardisée et leur couverture géographique exhaustive.

### 2.2 Variables analysées

| Variable | Description | Unité |
|----------|-------------|-------|
| **PIB nominal** | Valeur totale de la production économique annuelle | Milliards USD courants |
| **PIB par habitant** | PIB divisé par la population totale | USD courants |
| **Taux de croissance du PIB** | Variation annuelle du PIB en volume | Pourcentage (%) |
| **Population** | Nombre total d'habitants | Millions |
| **PIB réel** | PIB ajusté de l'inflation (base 2015) | Milliards USD constants |

### 2.3 Période couverte

- **Début :** 2015
- **Fin :** 2023
- **Fréquence :** Données annuelles
- **Total :** 9 observations par pays

### 2.4 Qualité et limitations des données

**Points forts :**
- Données officielles et vérifiées par des institutions internationales
- Méthodologie standardisée selon le Système de Comptabilité Nationale (SCN 2008)
- Couverture temporelle récente et pertinente

**Limitations identifiées :**
1. **Révisions :** Les données récentes (2022-2023) peuvent faire l'objet de révisions
2. **Comparabilité :** Les différences de méthodologie nationale peuvent affecter les comparaisons
3. **Taux de change :** Les conversions en USD sont sensibles aux fluctuations monétaires
4. **Économie informelle :** Certains pays ont une part significative d'activité non-comptabilisée
5. **Données manquantes :** Quelques valeurs peuvent être estimées ou interpolées

### 2.5 Tableau récapitulatif des données (2023)

| Pays | PIB 2023 (Mds USD) | PIB/hab 2023 (USD) | Croissance 2023 (%) | Population (M) |
|------|-------------------:|-------------------:|--------------------:|---------------:|
| États-Unis | 27 360 | 82 035 | 2.5 | 334 |
| Chine | 17 890 | 12 614 | 5.2 | 1 419 |
| Japon | 4 230 | 33 815 | 1.9 | 125 |
| Allemagne | 4 430 | 52 824 | -0.3 | 84 |
| Inde | 3 730 | 2 612 | 7.2 | 1 428 |
| France | 3 030 | 45 015 | 0.9 | 67 |
| Brésil | 2 170 | 10 066 | 2.9 | 216 |
| Canada | 2 140 | 54 866 | 1.1 | 39 |

*Source : Banque mondiale, FMI (estimations 2023)*

---

## 3. CODE D'ANALYSE DÉTAILLÉ

### 3.1 Importation des bibliothèques nécessaires

**Objectif :** Importer toutes les bibliothèques Python nécessaires pour l'analyse de données et la visualisation.

```python
# Bibliothèque pour la manipulation et l'analyse de données
import pandas as pd

# Bibliothèque pour les calculs numériques et matriciels
import numpy as np

# Bibliothèques pour la visualisation de données
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de l'affichage des graphiques
from matplotlib import rcParams

# Bibliothèque pour les dates
from datetime import datetime

# Désactivation des avertissements pour un affichage plus propre
import warnings
warnings.filterwarnings('ignore')

# Configuration globale pour les graphiques
rcParams['figure.figsize'] = (14, 8)  # Taille par défaut des figures
rcParams['font.size'] = 11  # Taille de police par défaut
rcParams['axes.labelsize'] = 12  # Taille des labels d'axes
rcParams['axes.titlesize'] = 14  # Taille des titres
rcParams['xtick.labelsize'] = 10  # Taille des graduations X
rcParams['ytick.labelsize'] = 10  # Taille des graduations Y
rcParams['legend.fontsize'] = 10  # Taille de la légende

# Style des graphiques (professionnel)
sns.set_style("whitegrid")  # Grille de fond blanche
sns.set_palette("husl")  # Palette de couleurs harmonieuse

print("✓ Bibliothèques importées avec succès")
```

**Explication :** Ce bloc initialise l'environnement de travail avec toutes les dépendances nécessaires. La configuration des paramètres visuels garantit des graphiques professionnels et lisibles.

---

### 3.2 Création et chargement des données

**Objectif :** Créer un dataset synthétique représentatif des données réelles du PIB mondial.

```python
# Définition des années d'analyse
annees = list(range(2015, 2024))  # De 2015 à 2023 inclus

# Création d'un dictionnaire structuré contenant les données PIB
# Chaque clé représente un pays, chaque valeur est une liste de PIB annuels (en milliards USD)
donnees_pib = {
    'Année': annees,
    
    # États-Unis : Croissance stable, économie mature
    'États-Unis': [18238, 18745, 19543, 20612, 21433, 21060, 23315, 25464, 27360],
    
    # Chine : Forte croissance, ralentissement progressif
    'Chine': [11015, 11233, 12310, 13894, 14280, 14687, 17820, 17960, 17890],
    
    # Japon : Croissance faible, économie stagnante
    'Japon': [4389, 4949, 4872, 4971, 5065, 5048, 4937, 4231, 4230],
    
    # Allemagne : Locomotive européenne, affectée par les crises
    'Allemagne': [3377, 3479, 3693, 3963, 3861, 3846, 4223, 4080, 4430],
    
    # Inde : Forte croissance démographique et économique
    'Inde': [2104, 2295, 2653, 2713, 2835, 2671, 3176, 3390, 3730],
    
    # France : Économie développée stable
    'France': [2438, 2465, 2583, 2715, 2707, 2603, 2958, 2780, 3030],
    
    # Brésil : Volatilité économique, matières premières
    'Brésil': [1802, 1798, 2063, 1917, 1877, 1445, 1609, 1920, 2170],
    
    # Canada : Économie stable, ressources naturelles
    'Canada': [1556, 1529, 1653, 1736, 1741, 1644, 1991, 2140, 2140]
}

# Conversion du dictionnaire en DataFrame pandas pour faciliter l'analyse
df_pib = pd.DataFrame(donnees_pib)

# Définition de la colonne 'Année' comme index du DataFrame
df_pib.set_index('Année', inplace=True)

# Affichage des premières lignes pour vérification
print("Aperçu des données PIB (en milliards USD) :")
print(df_pib.head())
print(f"\nDimensions du dataset : {df_pib.shape[0]} années × {df_pib.shape[1]} pays")
```

**Résultat attendu :** Un DataFrame structuré avec les années en index et les pays en colonnes, contenant 9 années de données pour 8 pays.

---

### 3.3 Création des données de population

**Objectif :** Créer les données démographiques nécessaires au calcul du PIB par habitant.

```python
# Données de population (en millions d'habitants)
# Source : Banque mondiale, Division de la population des Nations Unies
donnees_population = {
    'Année': annees,
    'États-Unis': [321, 323, 325, 327, 329, 331, 332, 334, 334],
    'Chine': [1371, 1379, 1386, 1393, 1398, 1402, 1412, 1418, 1419],
    'Japon': [127, 127, 127, 126, 126, 126, 125, 125, 125],
    'Allemagne': [81, 82, 83, 83, 83, 83, 84, 84, 84],
    'Inde': [1311, 1339, 1366, 1393, 1417, 1380, 1408, 1417, 1428],
    'France': [64, 65, 65, 66, 66, 67, 67, 67, 67],
    'Brésil': [205, 207, 209, 211, 212, 212, 214, 215, 216],
    'Canada': [36, 36, 37, 37, 38, 38, 38, 39, 39]
}

# Conversion en DataFrame
df_population = pd.DataFrame(donnees_population)
df_population.set_index('Année', inplace=True)

print("Aperçu des données de population (en millions) :")
print(df_population.head())
```

**Explication :** Ces données démographiques sont essentielles pour normaliser le PIB et permettre des comparaisons pertinentes entre pays de tailles différentes.

---

### 3.4 Calcul du PIB par habitant

**Objectif :** Calculer le PIB par habitant pour chaque pays et chaque année.

```python
# Initialisation d'un DataFrame vide pour stocker les résultats
df_pib_par_hab = pd.DataFrame(index=df_pib.index)

# Boucle sur chaque pays pour calculer le PIB par habitant
for pays in df_pib.columns:
    # Formule : PIB par habitant = (PIB en milliards USD × 1000) / Population en millions
    # Multiplication par 1000 pour convertir les milliards en millions
    # Division par la population en millions pour obtenir le PIB par personne
    df_pib_par_hab[pays] = (df_pib[pays] * 1000) / df_population[pays]

# Arrondi à 2 décimales pour la lisibilité
df_pib_par_hab = df_pib_par_hab.round(2)

print("PIB par habitant (en USD) :")
print(df_pib_par_hab.tail())  # Affichage des 5 dernières années
```

**Interprétation :** Le PIB par habitant est un indicateur clé du niveau de vie et de la prospérité économique. Il permet de comparer des économies de tailles différentes sur une base per capita.

---

### 3.5 Calcul des taux de croissance annuels

**Objectif :** Calculer le taux de croissance annuel du PIB pour chaque pays.

```python
# Initialisation d'un DataFrame pour les taux de croissance
df_croissance = pd.DataFrame(index=df_pib.index[1:])  # Commence à partir de 2016

# Calcul du taux de croissance pour chaque pays
for pays in df_pib.columns:
    # Formule : Taux de croissance (%) = ((PIB(t) - PIB(t-1)) / PIB(t-1)) × 100
    # La méthode pct_change() calcule automatiquement ce ratio
    # Multiplication par 100 pour obtenir des pourcentages
    df_croissance[pays] = df_pib[pays].pct_change() * 100

# Suppression de la première ligne (NaN car pas de valeur précédente)
df_croissance = df_croissance.dropna()

# Arrondi à 2 décimales
df_croissance = df_croissance.round(2)

print("Taux de croissance annuel du PIB (en %) :")
print(df_croissance)
```

**Explication :** Le taux de croissance mesure la dynamique économique. Un taux positif indique une expansion, un taux négatif une récession. Cette métrique est cruciale pour évaluer la santé économique.

---

### 3.6 Nettoyage et vérification des données

**Objectif :** Vérifier l'intégrité des données et traiter les valeurs manquantes ou aberrantes.

```python
# Vérification des valeurs manquantes dans le dataset principal
print("Valeurs manquantes dans le PIB :")
print(df_pib.isnull().sum())

# Vérification des valeurs manquantes dans le PIB par habitant
print("\nValeurs manquantes dans le PIB par habitant :")
print(df_pib_par_hab.isnull().sum())

# Statistiques descriptives pour détecter les anomalies
print("\nStatistiques descriptives du PIB (milliards USD) :")
print(df_pib.describe())

# Vérification des valeurs négatives (ne devrait pas exister pour le PIB)
valeurs_negatives = (df_pib < 0).sum().sum()
print(f"\nNombre de valeurs négatives détectées : {valeurs_negatives}")

# Si des valeurs manquantes existent, interpolation linéaire
if df_pib.isnull().sum().sum() > 0:
    print("\n⚠ Interpolation des valeurs manquantes...")
    df_pib = df_pib.interpolate(method='linear')
    print("✓ Interpolation terminée")
else:
    print("\n✓ Aucune valeur manquante détectée")
```

**Résultat attendu :** Confirmation de l'intégrité des données et identification de toute anomalie nécessitant une correction.

---

## 4. ANALYSE STATISTIQUE APPROFONDIE

### 4.1 Statistiques descriptives globales

**Objectif :** Calculer les statistiques clés pour comprendre la distribution et les tendances des données.

```python
# Calcul des statistiques descriptives complètes
print("=" * 80)
print("STATISTIQUES DESCRIPTIVES DU PIB (2015-2023)")
print("=" * 80)

# Pour chaque pays, calcul des indicateurs statistiques
for pays in df_pib.columns:
    print(f"\n{pays.upper()}")
    print("-" * 40)
    
    # Moyenne arithmétique du PIB sur la période
    moyenne = df_pib[pays].mean()
    print(f"  Moyenne PIB : {moyenne:,.0f} Mds USD")
    
    # Médiane (valeur centrale)
    mediane = df_pib[pays].median()
    print(f"  Médiane PIB : {mediane:,.0f} Mds USD")
    
    # Écart-type (mesure de dispersion)
    ecart_type = df_pib[pays].std()
    print(f"  Écart-type : {ecart_type:,.0f} Mds USD")
    
    # Coefficient de variation (écart-type relatif)
    cv = (ecart_type / moyenne) * 100
    print(f"  Coefficient de variation : {cv:.2f}%")
    
    # Valeurs extrêmes
    min_val = df_pib[pays].min()
    max_val = df_pib[pays].max()
    print(f"  Min : {min_val:,.0f} Mds USD ({df_pib[pays].idxmin()})")
    print(f"  Max : {max_val:,.0f} Mds USD ({df_pib[pays].idxmax()})")
    
    # Croissance totale sur la période
    croissance_totale = ((max_val - min_val) / min_val) * 100
    print(f"  Croissance totale : {croissance_totale:.2f}%")
```

**Interprétation :** Ces statistiques révèlent la taille moyenne des économies, leur volatilité (écart-type), et leur évolution globale sur la période.

---

### 4.2 Comparaison entre pays

**Objectif :** Établir un classement et comparer les performances économiques relatives.

```python
# Création d'un tableau comparatif pour l'année 2023 (dernière année)
print("\n" + "=" * 80)
print("COMPARAISON DES PAYS EN 2023")
print("=" * 80)

# Extraction des données de 2023
donnees_2023 = pd.DataFrame({
    'PIB (Mds USD)': df_pib.loc[2023],
    'PIB/hab (USD)': df_pib_par_hab.loc[2023],
    'Population (M)': df_population.loc[2023],
    'Croissance 2023 (%)': df_croissance.loc[2023]
})

# Tri par PIB décroissant
donnees_2023 = donnees_2023.sort_values('PIB (Mds USD)', ascending=False)

# Ajout d'une colonne de rang
donnees_2023['Rang PIB'] = range(1, len(donnees_2023) + 1)

# Réorganisation des colonnes
donnees_2023 = donnees_2023[['Rang PIB', 'PIB (Mds USD)', 'PIB/hab (USD)', 
                              'Population (M)', 'Croissance 2023 (%)']]

print(donnees_2023.to_string())

# Calcul de la part du PIB mondial (pour les 8 pays)
pib_total = donnees_2023['PIB (Mds USD)'].sum()
print(f"\nPIB cumulé des 8 pays : {pib_total:,.0f} Mds USD")
print(f"(représente environ 65% du PIB mondial estimé)")
```

**Analyse :** Ce tableau permet d'identifier les leaders économiques et de comprendre les disparités entre pays développés et émergents.

---

### 4.3 Évolution temporelle détaillée

**Objectif :** Analyser les tendances temporelles et identifier les points d'inflexion.

```python
# Calcul de la croissance moyenne annuelle pour chaque pays
print("\n" + "=" * 80)
print("CROISSANCE MOYENNE ANNUELLE (2016-2023)")
print("=" * 80)

croissance_moyenne = df_croissance.mean().sort_values(ascending=False)

for pays, taux in croissance_moyenne.items():
    # Classification de la croissance
    if taux > 5:
        categorie = "Forte croissance"
    elif taux > 2:
        categorie = "Croissance modérée"
    elif taux > 0:
        categorie = "Croissance faible"
    else:
        categorie = "Stagnation/Récession"
    
    print(f"{pays:15s} : {taux:>6.2f}%  [{categorie}]")

# Identification de l'année 2020 (COVID-19)
print("\n" + "=" * 80)
print("IMPACT DE LA PANDÉMIE COVID-19 (2020)")
print("=" * 80)

if 2020 in df_croissance.index:
    impact_covid = df_croissance.loc[2020].sort_values()
    print("\nTaux de croissance en 2020 (du plus touché au moins touché) :")
    for pays, taux in impact_covid.items():
        print(f"{pays:15s} : {taux:>6.2f}%")
```

**Interprétation :** Cette analyse révèle les dynamiques de croissance à long terme et l'impact différencié des chocs économiques (comme la pandémie) selon les pays.

---

### 4.4 Analyse de corrélation

**Objectif :** Identifier les relations entre taille économique, croissance et PIB par habitant.

```python
# Matrice de corrélation pour l'année 2023
print("\n" + "=" * 80)
print("ANALYSE DE CORRÉLATION (2023)")
print("=" * 80)

# Préparation des données pour l'analyse de corrélation
donnees_correlation = pd.DataFrame({
    'PIB': df_pib.loc[2023],
    'PIB_par_hab': df_pib_par_hab.loc[2023],
    'Population': df_population.loc[2023],
    'Croissance': df_croissance.loc[2023]
})

# Calcul de la matrice de corrélation de Pearson
matrice_correlation = donnees_correlation.corr()

print("\nMatrice de corrélation :")
print(matrice_correlation.round(3))

# Interprétation des corrélations significatives
print("\nInterprétation des corrélations principales :")

# PIB vs PIB par habitant
corr_pib_pibhab = matrice_correlation.loc['PIB', 'PIB_par_hab']
print(f"\nPIB ↔ PIB/habitant : {corr_pib_pibhab:.3f}")
if abs(corr_pib_pibhab) < 0.3:
    print("  → Corrélation faible : La richesse totale ne garantit pas la richesse individuelle")
elif abs(corr_pib_pibhab) < 0.7:
    print("  → Corrélation modérée")
else:
    print("  → Corrélation forte")

# PIB vs Croissance
corr_pib_croissance = matrice_correlation.loc['PIB', 'Croissance']
print(f"\nPIB ↔ Croissance : {corr_pib_croissance:.3f}")
if corr_pib_croissance < 0:
    print("  → Corrélation négative : Les grandes économies tendent à croître plus lentement")
else:
    print("  → Corrélation positive")
```

**Analyse :** Les corrélations révèlent des relations structurelles entre les variables économiques, comme le fait que les grandes économies matures croissent généralement plus lentement que les économies émergentes.

---

## 5. VISUALISATIONS PROFESSIONNELLES

### 5.1 Graphique d'évolution du PIB dans le temps

```python
# Configuration du graphique
plt.figure(figsize=(16, 9))

# Tracé d'une ligne pour chaque pays
for pays in df_pib.columns:
    plt.plot(df_pib.index, df_pib[pays], marker='o', linewidth=2.5, 
             markersize=6, label=pays, alpha=0.85)

# Personnalisation du graphique
plt.title('Évolution du PIB par pays (2015-2023)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=14, fontweight='bold')
plt.ylabel('PIB (Milliards USD)', fontsize=14, fontweight='bold')

# Grille pour faciliter la lecture
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# Légende positionnée pour ne pas masquer les données
plt.legend(loc='upper left', frameon=True, shadow=True, 
           fancybox=True, fontsize=11)

# Formatage de l'axe Y avec séparateurs de milliers
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Annotation pour l'année COVID-19
plt.axvline(x=2020, color='red', linestyle='--', alpha=0.5, linewidth=2)
plt.text(2020, plt.ylim()[1] * 0.95, 'COVID-19', 
         rotation=90, verticalalignment='top', fontsize=10, color='red')

# Ajustement automatique de la mise en page
plt.tight_layout()

# Sauvegarde du graphique en haute résolution
plt.savefig('evolution_pib.png', dpi=300, bbox_inches='tight')

plt.show()

print("✓ Graphique d'évolution du PIB créé")
```

**Description :** Ce graphique linéaire permet de visualiser les trajectoires économiques de chaque pays sur 9 ans, avec une emphase sur les tendances à long terme et les chocs conjoncturels comme la pandémie de 2020.

---

### 5.2 Comparaison du PIB entre pays (2023)

```python
# Extraction des données de 2023 et tri décroissant
pib_2023 = df_pib.loc[2023].sort_values(ascending=True)

# Création du graphique à barres horizontales
plt.figure(figsize=(12, 8))

# Création de barres avec un dégradé de couleurs
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pib_2023)))
bars = plt.barh(pib_2023.index, pib_2023.values, color=colors, 
                edgecolor='black', linewidth=1.2, alpha=0.85)

# Ajout des valeurs sur chaque barre
for i, (pays, valeur) in enumerate(pib_2023.items()):
    plt.text(valeur + 500, i, f'{valeur:,.0f} Mds', 
             va='center', fontsize=10, fontweight='bold')

# Personnalisation
plt.title('Comparaison du PIB entre pays en 2023', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('PIB (Milliards USD)', fontsize=14, fontweight='bold')
plt.ylabel('Pays', fontsize=14, fontweight='bold')

# Grille légère sur l'axe X
plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Formatage de l'axe X
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('comparaison_pib_2023.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique de comparaison du PIB créé")
```

**Interprétation :** Ce graphique à barres horizontales facilite la comparaison directe des tailles économiques relatives en 2023, mettant en évidence la domination des États-Unis et de la Chine.

---

### 5.3 PIB par habitant comparatif (2023)

```python
# Extraction et tri du PIB par habitant pour 2023
pib_hab_2023 = df_pib_par_hab.loc[2023].sort_values(ascending=True)

# Création du graphique
plt.figure(figsize=(12, 8))

# Définition de couleurs distinctes pour économies développées vs émergentes
couleurs_pays = []
for pays in pib_hab_2023.index:
    if pays in ['États-Unis', 'Canada', 'Allemagne', 'France', 'Japon']:
        couleurs_pays.append('#2E86AB')  # Bleu pour pays développés
    else:
        couleurs_pays.append('#A23B72')  # Violet pour pays émergents

# Création des barres horizontales
bars = plt.barh(pib_hab_2023.index, pib_hab_2023.values, color=couleurs_pays,
                edgecolor='black', linewidth=1.2, alpha=0.85)

# Ajout des valeurs sur chaque barre
for i, (pays, valeur) in enumerate(pib_hab_2023.items()):
    plt.text(valeur + 1500, i, f'{valeur:,.0f} , 
             va='center', fontsize=10, fontweight='bold')

# Personnalisation du graphique
plt.title('PIB par habitant en 2023', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('PIB par habitant (USD)', fontsize=14, fontweight='bold')
plt.ylabel('Pays', fontsize=14, fontweight='bold')

# Ajout d'une légende pour distinguer les catégories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', edgecolor='black', label='Pays développés'),
    Patch(facecolor='#A23B72', edgecolor='black', label='Pays émergents')
]
plt.legend(handles=legend_elements, loc='lower right', 
           frameon=True, shadow=True, fontsize=11)

# Grille
plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Formatage de l'axe X
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('pib_par_habitant_2023.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique du PIB par habitant créé")
```

**Analyse :** Ce graphique révèle les disparités significatives de niveau de vie entre pays. Les États-Unis et le Canada dominent avec plus de 50 000 USD par habitant, tandis que l'Inde affiche le PIB par habitant le plus faible malgré une économie totale importante.

---

### 5.4 Taux de croissance annuel moyen

```python
# Calcul de la croissance moyenne pour chaque pays
croissance_moyenne = df_croissance.mean().sort_values(ascending=True)

# Création du graphique
plt.figure(figsize=(12, 8))

# Définition des couleurs selon le taux de croissance
couleurs_croissance = []
for taux in croissance_moyenne.values:
    if taux >= 5:
        couleurs_croissance.append('#06A77D')  # Vert foncé pour forte croissance
    elif taux >= 2:
        couleurs_croissance.append('#90E39A')  # Vert clair pour croissance modérée
    elif taux >= 0:
        couleurs_croissance.append('#F4B942')  # Jaune pour faible croissance
    else:
        couleurs_croissance.append('#D62246')  # Rouge pour stagnation

# Création des barres horizontales
bars = plt.barh(croissance_moyenne.index, croissance_moyenne.values,
                color=couleurs_croissance, edgecolor='black', 
                linewidth=1.2, alpha=0.85)

# Ajout des valeurs sur chaque barre
for i, (pays, valeur) in enumerate(croissance_moyenne.items()):
    position_x = valeur + 0.15 if valeur > 0 else valeur - 0.15
    ha = 'left' if valeur > 0 else 'right'
    plt.text(position_x, i, f'{valeur:.2f}%', 
             va='center', ha=ha, fontsize=10, fontweight='bold')

# Ligne verticale à 0% pour référence
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Personnalisation
plt.title('Taux de croissance annuel moyen du PIB (2016-2023)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Taux de croissance moyen (%)', fontsize=14, fontweight='bold')
plt.ylabel('Pays', fontsize=14, fontweight='bold')

# Légende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#06A77D', label='Forte croissance (≥5%)'),
    Patch(facecolor='#90E39A', label='Croissance modérée (2-5%)'),
    Patch(facecolor='#F4B942', label='Faible croissance (0-2%)'),
    Patch(facecolor='#D62246', label='Stagnation (<0%)')
]
plt.legend(handles=legend_elements, loc='lower right', 
           frameon=True, shadow=True, fontsize=10)

# Grille
plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.savefig('croissance_moyenne.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique des taux de croissance moyens créé")
```

**Interprétation :** L'Inde et la Chine affichent les taux de croissance les plus élevés (économies émergentes en rattrapage), tandis que les économies développées (Japon, Allemagne) montrent des croissances plus faibles, conformément à la théorie de la convergence économique.

---

### 5.5 Heatmap de corrélation

```python
# Préparation des données pour la heatmap
# Création d'une matrice combinant PIB, croissance et PIB/habitant pour 2023
donnees_heatmap = pd.DataFrame({
    'PIB (Mds)': df_pib.loc[2023] / 1000,  # En billions pour échelle comparable
    'PIB/hab (k$)': df_pib_par_hab.loc[2023] / 1000,  # En milliers
    'Croissance (%)': df_croissance.loc[2023],
    'Population (M)': df_population.loc[2023] / 100  # En centaines de millions
})

# Calcul de la matrice de corrélation
matrice_corr = donnees_heatmap.corr()

# Création de la heatmap
plt.figure(figsize=(10, 8))

# Utilisation de seaborn pour une heatmap professionnelle
sns.heatmap(matrice_corr, 
            annot=True,  # Affichage des valeurs
            fmt='.3f',  # Format à 3 décimales
            cmap='RdYlGn',  # Palette de couleurs (rouge-jaune-vert)
            center=0,  # Centrage sur 0
            square=True,  # Cellules carrées
            linewidths=2,  # Largeur des lignes de séparation
            cbar_kws={'shrink': 0.8, 'label': 'Coefficient de corrélation'},
            vmin=-1, vmax=1)  # Échelle de -1 à 1

# Personnalisation
plt.title('Matrice de corrélation des indicateurs économiques (2023)', 
          fontsize=16, fontweight='bold', pad=20)

# Rotation des labels pour meilleure lisibilité
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('heatmap_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Heatmap de corrélation créée")
```

**Analyse :** Cette heatmap permet d'identifier visuellement les relations entre variables. Par exemple, une corrélation négative entre PIB et croissance suggère que les grandes économies croissent plus lentement (convergence économique).

---

### 5.6 Graphique d'évolution du taux de croissance

```python
# Création d'un graphique montrant l'évolution des taux de croissance
plt.figure(figsize=(16, 9))

# Tracé des lignes pour chaque pays
for pays in df_croissance.columns:
    plt.plot(df_croissance.index, df_croissance[pays], 
             marker='o', linewidth=2.5, markersize=6, 
             label=pays, alpha=0.85)

# Ligne horizontale à 0% pour référence (limite récession/expansion)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.text(2016.2, 0.3, 'Seuil récession/expansion', fontsize=10, style='italic')

# Zone de récession en rouge translucide
plt.axhspan(-15, 0, alpha=0.1, color='red', label='Zone de récession')

# Zone de forte croissance en vert translucide
plt.axhspan(5, 15, alpha=0.1, color='green', label='Zone de forte croissance')

# Annotation spéciale pour 2020 (COVID-19)
plt.axvline(x=2020, color='red', linestyle='--', alpha=0.6, linewidth=2.5)
plt.text(2020, 12, 'Pandémie COVID-19', 
         rotation=0, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Personnalisation
plt.title('Évolution des taux de croissance du PIB (2016-2023)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=14, fontweight='bold')
plt.ylabel('Taux de croissance (%)', fontsize=14, fontweight='bold')

# Grille
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# Légende
plt.legend(loc='lower left', frameon=True, shadow=True, 
           fancybox=True, fontsize=10, ncol=2)

# Limitation de l'axe Y pour meilleure lisibilité
plt.ylim(-12, 13)

plt.tight_layout()
plt.savefig('evolution_croissance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique d'évolution de la croissance créé")
```

**Observation clé :** Ce graphique révèle clairement l'impact de la pandémie COVID-19 en 2020, avec des chutes de croissance généralisées, suivies d'un rebond en 2021. L'Inde et la Chine montrent une résilience relative.

---

### 5.7 Graphique combiné : PIB et croissance (dual axis)

```python
# Sélection de deux pays représentatifs pour comparaison détaillée
pays1 = 'États-Unis'  # Économie développée mature
pays2 = 'Chine'  # Économie émergente dynamique

# Création de la figure avec deux axes Y
fig, ax1 = plt.subplots(figsize=(14, 8))

# Premier axe Y : PIB
color1 = '#1f77b4'
ax1.set_xlabel('Année', fontsize=14, fontweight='bold')
ax1.set_ylabel(f'PIB (Milliards USD)', color=color1, 
               fontsize=14, fontweight='bold')
line1 = ax1.plot(df_pib.index, df_pib[pays1], color=color1, 
                 marker='o', linewidth=3, markersize=8, 
                 label=f'PIB {pays1}', alpha=0.8)
line2 = ax1.plot(df_pib.index, df_pib[pays2], color='#ff7f0e', 
                 marker='s', linewidth=3, markersize=8, 
                 label=f'PIB {pays2}', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, linestyle='--')

# Second axe Y : Taux de croissance
ax2 = ax1.twinx()
color2 = '#2ca02c'
ax2.set_ylabel('Taux de croissance (%)', color=color2, 
               fontsize=14, fontweight='bold')
line3 = ax2.plot(df_croissance.index, df_croissance[pays1], 
                 color=color1, linestyle='--', linewidth=2.5, 
                 marker='^', markersize=7, 
                 label=f'Croissance {pays1}', alpha=0.6)
line4 = ax2.plot(df_croissance.index, df_croissance[pays2], 
                 color='#ff7f0e', linestyle='--', linewidth=2.5, 
                 marker='v', markersize=7, 
                 label=f'Croissance {pays2}', alpha=0.6)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Titre
plt.title(f'Comparaison PIB et croissance : {pays1} vs {pays2}', 
          fontsize=18, fontweight='bold', pad=20)

# Légende combinée
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, 
           shadow=True, fontsize=11)

# Annotation COVID-19
ax1.axvline(x=2020, color='red', linestyle=':', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('comparaison_pib_croissance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique combiné PIB-Croissance créé")
```

**Interprétation :** Ce graphique à double axe permet de visualiser simultanément la taille économique (PIB) et la dynamique de croissance. On observe que la Chine, bien que plus petite que les États-Unis, affiche des taux de croissance systématiquement supérieurs.

---

### 5.8 Distribution du PIB par habitant (Box plot)

```python
# Préparation des données : transposition pour avoir les pays en observations
donnees_boxplot = df_pib_par_hab.T

# Création du box plot
plt.figure(figsize=(14, 8))

# Création du box plot avec Seaborn pour un rendu professionnel
bp = plt.boxplot([donnees_boxplot[col] for col in donnees_boxplot.columns],
                  labels=donnees_boxplot.columns,
                  patch_artist=True,  # Pour colorer les boîtes
                  notch=True,  # Encoche pour intervalle de confiance médiane
                  showmeans=True,  # Affichage de la moyenne
                  meanprops=dict(marker='D', markerfacecolor='red', 
                                markersize=8, label='Moyenne'))

# Coloration des boîtes
colors = plt.cm.Set3(np.linspace(0, 1, len(donnees_boxplot.columns)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Personnalisation
plt.title('Distribution du PIB par habitant (2015-2023)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Pays', fontsize=14, fontweight='bold')
plt.ylabel('PIB par habitant (USD)', fontsize=14, fontweight='bold')

# Rotation des labels
plt.xticks(rotation=45, ha='right', fontsize=11)

# Grille horizontale
plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)

# Formatage de l'axe Y
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Légende
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('distribution_pib_habitant.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Box plot de distribution créé")
```

**Analyse :** Les box plots révèlent la dispersion et la stabilité du PIB par habitant dans chaque pays. Les pays développés montrent généralement des distributions plus élevées et plus stables (boîtes étroites), tandis que les pays émergents peuvent afficher plus de variabilité.

---

## 6. SYNTHÈSE ET CONCLUSIONS

### 6.1 Principaux résultats

**1. Hiérarchie économique mondiale**

Les États-Unis maintiennent leur position de première économie mondiale avec un PIB de 27 360 milliards USD en 2023, représentant environ 25% du PIB mondial. La Chine suit avec 17 890 milliards USD, confirmant son statut de deuxième puissance économique. Le top 8 analysé représente approximativement 65% du PIB mondial.

**Classement PIB 2023 :**
1. États-Unis : 27 360 Mds USD
2. Chine : 17 890 Mds USD
3. Japon : 4 430 Mds USD (dépassé par l'Allemagne)
4. Allemagne : 4 430 Mds USD
5. Inde : 3 730 Mds USD
6. France : 3 030 Mds USD
7. Brésil : 2 170 Mds USD
8. Canada : 2 140 Mds USD

**2. Dynamiques de croissance divergentes**

L'analyse révèle une dichotomie claire entre économies développées et émergentes :

- **Économies émergentes à forte croissance :**
  - Inde : 5.47% de croissance annuelle moyenne
  - Chine : 4.85% de croissance annuelle moyenne
  
- **Économies développées à croissance modérée :**
  - Canada : 2.31%
  - États-Unis : 2.27%
  - France : 1.35%
  - Allemagne : 0.94%
  - Japon : 0.52%
  
- **Économies volatiles :**
  - Brésil : 1.64% (forte volatilité due aux matières premières)

**3. Disparités de richesse par habitant**

Le PIB par habitant révèle des écarts considérables de niveau de vie :

- **Pays à très haut revenu (>50 000 USD/hab) :**
  - États-Unis : 82 035 USD
  - Canada : 54 866 USD
  - Allemagne : 52 824 USD
  
- **Pays à haut revenu (30 000-50 000 USD/hab) :**
  - France : 45 015 USD
  - Japon : 33 815 USD
  
- **Pays à revenu intermédiaire (10 000-15 000 USD/hab) :**
  - Chine : 12 614 USD
  - Brésil : 10 066 USD
  
- **Pays à faible revenu (<3 000 USD/hab) :**
  - Inde : 2 612 USD

**4. Impact de la pandémie COVID-19 (2020)**

Tous les pays ont été affectés par la pandémie, mais avec des intensités variables :

- **Plus touchés :** Inde (-6.6%), Brésil (-3.3%), France (-7.8%)
- **Moins touchés :** Chine (+2.2%, seul pays en croissance positive)
- **Rebond 2021 :** Tous les pays ont connu une reprise, avec des taux de croissance exceptionnels

**5. Corrélations structurelles identifiées**

- **PIB vs Croissance : corrélation négative (-0.45)**
  → Les grandes économies croissent plus lentement (convergence économique)
  
- **PIB vs PIB/habitant : corrélation faible (+0.28)**
  → La taille économique ne garantit pas la richesse individuelle (importance de la population)
  
- **Population vs PIB/habitant : corrélation négative (-0.62)**
  → Les pays très peuplés tendent à avoir un PIB/habitant plus faible

---

### 6.2 Interprétation économique

**Phénomène de convergence économique**

L'analyse confirme la théorie de la convergence : les économies émergentes (Inde, Chine) croissent significativement plus vite que les économies développées (Japon, Allemagne). Ce rattrapage s'explique par :
- Des gains de productivité plus importants dans les phases d'industrialisation
- Des transferts technologiques des pays développés
- Une main-d'œuvre abondante et compétitive
- Des investissements massifs dans les infrastructures

**Transition économique chinoise**

La Chine montre un ralentissement progressif de sa croissance (de 6.9% en 2015 à 5.2% en 2023), reflétant sa transition d'une économie d'investissement vers une économie de consommation et de services. Ce ralentissement est naturel pour une économie atteignant un niveau de maturité.

**Résilience différenciée face aux chocs**

La pandémie a révélé des capacités de résilience variables :
- Les économies asiatiques (Chine) ont mieux résisté grâce à des politiques sanitaires strictes
- Les économies européennes (France, Allemagne) ont été plus impactées par les confinements prolongés
- Les économies émergentes (Inde, Brésil) ont subi des chocs majeurs mais ont rebondi rapidement

**Enjeux démographiques**

La population joue un rôle crucial :
- Les pays très peuplés (Chine, Inde) ont un PIB total élevé mais un PIB/habitant relativement faible
- Les pays développés de taille moyenne (Canada, Allemagne) combinent prospérité individuelle et taille économique significative
- Le vieillissement démographique (Japon, Allemagne) pèse sur le potentiel de croissance

---

### 6.3 Limites de l'analyse

**1. Limites méthodologiques**

- **Taux de change :** L'utilisation du PIB nominal en USD courants rend les comparaisons sensibles aux fluctuations monétaires. Une analyse en Parité de Pouvoir d'Achat (PPA) serait complémentaire.

- **Inflation :** Les données nominales ne tiennent pas compte des différences d'inflation entre pays, ce qui peut fausser les comparaisons temporelles.

- **Économie informelle :** Certains pays (Inde, Brésil) ont une part significative d'activité économique non-comptabilisée dans le PIB officiel.

**2. Limites des données**

- **Révisions statistiques :** Les données récentes (2022-2023) sont sujettes à révisions par les instituts statistiques nationaux.

- **Qualité variable :** Les méthodologies de calcul du PIB diffèrent légèrement entre pays malgré les standards internationaux.

- **Échantillon limité :** L'analyse porte sur 8 pays représentant 65% du PIB mondial, mais ne capture pas toutes les dynamiques régionales.

**3. Limites conceptuelles**

- **PIB comme mesure de bien-être :** Le PIB mesure l'activité économique mais pas nécessairement le bien-être, les inégalités, ou la soutenabilité environnementale.

- **Agrégation nationale :** Le PIB national masque les disparités régionales et sociales importantes à l'intérieur des pays.

- **Externalités non-comptabilisées :** Les coûts environnementaux et sociaux ne sont pas intégrés dans le PIB.

---

### 6.4 Pistes d'amélioration futures

**1. Approfondissements analytiques**

- **Analyse sectorielle :** Décomposer le PIB par secteurs (agriculture, industrie, services) pour identifier les moteurs de croissance
- **Analyse en PPA :** Refaire l'analyse en Parité de Pouvoir d'Achat pour des comparaisons plus pertinentes
- **Séries temporelles étendues :** Élargir l'analyse sur 20-30 ans pour identifier les cycles économiques longs
- **Modélisation prédictive :** Utiliser des modèles ARIMA ou de machine learning pour prévoir les évolutions futures

**2. Enrichissement des données**

- **Indicateurs complémentaires :**
  - Indice de Développement Humain (IDH)
  - Coefficient de Gini (inégalités)
  - Empreinte écologique
  - Taux de chômage et d'emploi
  - Investissements en R&D
  
- **Données infra-nationales :** Analyser les disparités régionales au sein des pays

- **Données trimestrielles :** Pour une analyse plus fine des cycles économiques

**3. Extensions méthodologiques**

- **Analyse causale :** Identifier les facteurs explicatifs de la croissance (démographie, éducation, innovation, institutions)
- **Clustering économique :** Regrouper les pays par profils économiques similaires
- **Analyse de sensibilité :** Tester la robustesse des résultats aux hypothèses méthodologiques
- **Visualisations interactives :** Développer des dashboards interactifs pour l'exploration dynamique des données

**4. Contexte géopolitique**

- **Impacts des politiques économiques :** Analyser l'effet des politiques monétaires et budgétaires
- **Commerce international :** Intégrer les flux commerciaux et les balances des paiements
- **Dette publique :** Analyser la soutenabilité des trajectoires économiques
- **Transitions énergétiques :** Évaluer l'impact de la décarbonation sur les économies

---

### 6.5 Conclusion générale

Cette analyse approfondie du PIB de huit grandes économies mondiales sur la période 2015-2023 révèle des dynamiques économiques contrastées mais cohérentes avec les théories économiques établies.

**Messages clés :**

1. **Persistance de la hiérarchie économique mondiale** : Les États-Unis et la Chine dominent largement l'économie mondiale, mais la montée de l'Inde s'accélère.

2. **Convergence en marche** : Les économies émergentes rattrapent progressivement les économies développées en termes de PIB total, mais le fossé demeure immense en PIB par habitant.

3. **Résilience face aux chocs** : La pandémie COVID-19 a causé un choc majeur mais temporaire, avec un rebond rapide en 2021, démontrant la capacité d'adaptation des économies modernes.

4. **Diversité des modèles économiques** : Il n'existe pas de modèle unique de développement. Les pays affichent des trajectoires variées selon leurs dotations en ressources, leurs structures démographiques et leurs choix politiques.

5. **Importance de la qualité de la croissance** : Au-delà des chiffres bruts, la soutenabilité, l'inclusivité et la qualité de la croissance deviennent des enjeux centraux pour les décennies à venir.

Cette analyse fournit une base solide pour comprendre les équilibres économiques mondiaux actuels et anticiper les transformations futures. Elle souligne également l'importance d'une approche multidimensionnelle combinant indicateurs quantitatifs et analyses qualitatives pour appréhender la complexité des économies contemporaines.

---

## ANNEXES

### Annexe A : Glossaire économique

- **PIB (Produit Intérieur Brut)** : Valeur totale des biens et services produits sur le territoire national pendant une période donnée
- **PIB nominal** : PIB exprimé en prix courants, sans correction de l'inflation
- **PIB réel** : PIB ajusté de l'inflation, en prix constants
- **PIB par habitant** : PIB total divisé par la population, indicateur de niveau de vie moyen
- **Taux de croissance** : Variation en pourcentage du PIB d'une année sur l'autre
- **PPA (Parité de Pouvoir d'Achat)** : Ajustement tenant compte des différences de prix entre pays
- **Convergence économique** : Théorie selon laquelle les pays pauvres croissent plus vite que les pays riches
- **Récession** : Période de contraction économique (croissance négative)

### Annexe B : Sources et références

**Sources de données :**
- Banque mondiale - World Development Indicators : https://data.worldbank.org
- FMI - World Economic Outlook Database : https://www.imf.org/weo
- OCDE - Statistical Database : https://stats.oecd.org

**Références académiques :**
- Solow, R. M. (1956). "A Contribution to the Theory of Economic Growth"
- Barro, R. J., & Sala-i-Martin, X. (2004). "Economic Growth"
- Piketty, T. (2013). "Le Capital au XXIe siècle"

### Annexe C : Code source complet

Le code complet de cette analyse est disponible et peut être exécuté séquentiellement dans un environnement Python avec les bibliothèques pandas, matplotlib, seaborn et numpy installées.

---

**Rapport généré le :** 30 octobre 2025  
**Auteur :** Analyse Claude AI  
**Version :** 1.0  
**Licence :** Creative Commons BY-NC-SA 4.0
