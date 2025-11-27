# 🧾 Compte-Rendu d'Étude – Analyse de Sentiment de Posts Reddit sur des Artistes  

---

# 📌 Sommaire
1. [Introduction](#introduction)
2. [Objectifs](#objectifs)
3. [Description du dataset](#description-du-dataset)
4. [Méthodologie](#méthodologie)
5. [Analyse exploratoire](#analyse-exploratoire)
   - [Répartition des sentiments](#répartition-des-sentiments)
   - [Longueur des textes](#longueur-des-textes)
   - [Exemples par sentiment](#exemples-par-sentiment)
6. [Visualisations & Graphiques](#visualisations--graphiques)
   - [Graphique 1 – Distribution des sentiments](#graphique-1--distribution-des-sentiments)
   - [Graphique 2 – Distribution de la longueur des posts](#graphique-2--distribution-de-la-longueur-des-posts)
   - [Graphique 3 – Longueur moyenne par sentiment](#graphique-3--longueur-moyenne-par-sentiment)
7. [Résultats](#résultats)
8. [Limites](#limites)
9. [Conclusion](#conclusion)

---

# ⭐ Introduction

Ce rapport présente une analyse exploratoire d’un dataset de **posts Reddit sur des artistes**, annotés par **sentiment**.

Le fichier utilisé est :  

- `reddit_artist_posts_sentiment.csv`

Chaque ligne correspond à un post textuel, associé à un label de sentiment (`positive`, `negative`, `neutral`).  
L’objectif est de décrire la structure des données, analyser la distribution des sentiments et la longueur des messages, et donner des pistes pour un futur modèle de classification.

---

# 🎯 Objectifs

Les objectifs de cette étude sont :

- Décrire la **répartition des sentiments** dans le corpus.
- Analyser la **longueur des posts** (en caractères et en mots).
- Observer les différences de longueur selon le **type de sentiment**.
- Fournir des **exemples concrets** pour chaque sentiment.
- Préparer le terrain pour un **éventuel modèle de classification de sentiment**.

---

# 🗂 Description du dataset

Après chargement du fichier CSV, on obtient :

- **Nombre de lignes** : `31 948`
- **Nombre de colonnes** : `2`

Les colonnes sont :

- `text` : le contenu textuel du post Reddit  
- `label` : le sentiment associé au post (`positive`, `negative`, `neutral`)

---

# 🧪 Méthodologie

Les principales étapes d’analyse ont été :

1. **Chargement des données**  
   - Lecture du CSV avec `pandas`.

2. **Nettoyage léger**
   - Conversion systématique de `text` en chaîne de caractères.
   - Création de deux nouvelles colonnes :
     - `char_len` : longueur du texte en caractères
     - `word_len` : longueur du texte en nombre de mots

3. **Statistiques descriptives**
   - Répartition des labels (`value_counts`)
   - Statistiques sur les longueurs de texte (min, max, moyenne, quartiles)
   - Moyenne de longueur par sentiment

4. **Préparation des visualisations (à générer en Python ou autre)**
   - Histogrammes et barplots
   - Comparaison visuelle entre catégories de sentiments

---

# 📊 Analyse exploratoire

## 📌 Répartition des sentiments

Le dataset contient **31 948 posts**, répartis comme suit :

| Sentiment | Nombre de posts | Pourcentage approximatif |
|----------|-----------------|--------------------------|
| neutral  | 19 728          | 61,75 %                  |
| positive | 8 825           | 27,62 %                  |
| negative | 3 395           | 10,63 %                  |

### 🔍 Interprétation

- La majorité des posts sont **neutres** (~62 %) :  
  cela reflète probablement des messages factuels (annonces, critiques modérées, news).
- Les posts **positifs** représentent environ **28 %** du corpus.
- Les posts **négatifs** sont minoritaires (~11 %), ce qui crée un **déséquilibre de classes** à prendre en compte si on entraîne un modèle de Machine Learning (risque de biais vers la classe neutre).

---

## 📏 Longueur des textes

Deux métriques ont été calculées :

- `char_len` : longueur du texte en **caractères**
- `word_len` : longueur du texte en **mots**

### 📐 Statistiques globales (tous sentiments confondus)

**Longueur en caractères (`char_len`) :**

- Moyenne ≈ **96,3** caractères  
- Écart-type ≈ **61,0**  
- Minimum = **1**  
- Médiane ≈ **79**  
- Maximum = **280**

**Longueur en mots (`word_len`) :**

- Moyenne ≈ **16,8** mots  
- Écart-type ≈ **11,1**  
- Minimum = **1**  
- Médiane ≈ **13**  
- Maximum = **62**

### 🔍 Interprétation

- Les posts sont généralement **courts à moyens** (autour de 80–100 caractères / 13–17 mots).
- Quelques posts sont très longs (jusqu’à **62 mots**), ce qui peut être le cas de critiques détaillées ou longues discussions.
- La présence de textes très courts (1 mot) peut venir de réponses courtes, titres, ou posts minimalistes.

---

## 📏 Longueur moyenne par sentiment

La longueur moyenne varie selon le sentiment :

| Sentiment | Longueur moyenne (caractères) | Longueur moyenne (mots) |
|----------|-------------------------------|--------------------------|
| negative | 116,2                          | 20,5                     |
| positive | 112,0                          | 19,9                     |
| neutral  | 85,8                           | 14,8                     |

### 🔍 Interprétation

- Les posts **négatifs** et **positifs** sont **plus longs en moyenne** que les posts neutres.  
- Les messages neutres sont souvent plus **factuels** ou concis (annonces, infos brutes).
- Les messages avec un **sentiment fort** (positif ou négatif) ont tendance à être plus détaillés :  
  explications, justifications, avis nuancés.

---

## 💬 Exemples par sentiment

Voici quelques exemples réels issus du dataset (tronqués si besoin) :

### 😡 Exemple de post *négatif* :

> `pitchfork track review: taylor swift’s “actually romantic” is actually embarrassing`

→ Ton négatif, jugement critique sur un morceau.

---

### 😃 Exemple de post *positif* :

> `taylor swift has regained the masters of her first six albums.`

→ Ton positif, bonne nouvelle, formulation factuelle mais connotée positivement.

---

### 😐 Exemple de post *neutre* :

> `pitchfork review: taylor swift - the life of a showgirl (5.9)`

→ Plutôt descriptif, neutre, annonce d’une review et d’une note.

---

# 🖼️ Visualisations & Graphiques

> 💡 Les chemins d’images ci-dessous supposent que tu sauvegardes tes figures dans un dossier `images/` à la racine du repo.

---

## 📊 Graphique 1 – Distribution des sentiments

```
python
# Exemple de code pour générer la figure
import matplotlib.pyplot as plt

counts = df['label'].value_counts()

plt.figure()
counts.plot(kind='bar')
plt.xlabel("Sentiment")
plt.ylabel("Nombre de posts")
plt.title("Distribution des sentiments")

plt.figure()
df['word_len'].hist(bins=30)
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de posts")
plt.title("Distribution de la longueur des posts (en mots)")
plt.tight_layout()
plt.savefig("images/text_length_distribution.png")
```
🔍 Analyse

Le pic principal se situe autour de 10–20 mots, confirmant que la plupart des posts sont assez courts.

La queue de distribution montre l’existence de posts beaucoup plus longs :
ces posts peuvent contenir des avis plus développés, critiques détaillées ou débats.

```
avg_len = df.groupby('label')['word_len'].mean().loc[['negative','neutral','positive']]

plt.figure()
avg_len.plot(kind='bar')
plt.ylabel("Longueur moyenne (mots)")
plt.xlabel("Sentiment")
plt.title("Longueur moyenne des posts par sentiment")
plt.tight_layout()
plt.savefig("images/avg_length_by_sentiment.png")

```
🔍 Analyse

Les posts négatifs sont légèrement les plus longs, suivis par les positifs.

Les posts neutres sont significativement plus courts.

Cela confirme l’hypothèse : plus l’auteur exprime une émotion ou un avis, plus il écrit de texte.

# ✅ Résultats

Les principaux résultats de cette analyse sont :

## 📌 Répartition des sentiments
- **~62 % neutres**
- **~28 % positifs**
- **~11 % négatifs**

➡️ Le dataset est **déséquilibré**, ce qui doit être pris en compte pour entraîner un modèle de Machine Learning.

---

## 📏 Longueur des posts
- En moyenne : **~17 mots**
- Variance importante entre les posts
- Les posts peuvent être :
  - **très courts** : 1 mot  
  - **assez longs** : jusqu’à 62 mots  

---

## 🔍 Différences par sentiment
- Les posts **positifs** et **négatifs** sont **plus longs** que les neutres.
- Les posts **neutres** ont tendance à être plus **factuels** et concis.
- Les messages exprimant une émotion forte (±) sont plus détaillés.

---

## 💬 Exemples concrets
Les exemples extraits du dataset confirment l’intuition :

- Les posts **négatifs** expriment des critiques détaillées.
- Les posts **positifs** expriment de bonnes nouvelles ou du soutien.
- Les posts **neutres** sont des informations factuelles (annonces, revues, notes).

---

# ⚠️ Limites

- Analyse basée uniquement sur la **dimension textuelle** (pas d’informations sur :
  - auteur  
  - date  
  - subreddit  
  - karma, etc.)
- Dataset **déséquilibré** → risque de biais en classification.
- Les labels *positive / negative / neutral* sont supposés corrects, mais il peut exister du **bruit d’annotation**.
- Non inclus dans ce rapport :
  - Analyse linguistique avancée (n-grams, vocabulaire)
  - Entraînement d’un modèle de classification

---

# 🏁 Conclusion

Cette première analyse exploratoire du dataset `reddit_artist_posts_sentiment.csv` montre que :

- La **répartition des sentiments** est fortement déséquilibrée.
- La **longueur des posts** varie selon le sentiment.
- Les posts exprimant une émotion (positive ou négative) sont **plus longs** et **plus développés**.
- Le dataset présente des caractéristiques importantes pour la mise en place d’un futur modèle NLP.

---

# 👣 Prochaines étapes possibles

### 🔧 Extraction de features
- TF-IDF  
- Bag-of-Words  
- Word embeddings (Word2Vec, GloVe)  
- Transformers embeddings (BERT, RoBERTa)

### 🤖 Modélisation
- Logistic Regression  
- SVM  
- Random Forest  
- BERT finetuné  

### 📊 Évaluation (dataset déséquilibré)
- F1-score par classe  
- Macro-F1  
- Matrice de confusion  
- Balanced Accuracy  

### 📚 Analyse complémentaire
- Wordclouds par sentiment  
- Top n-grams  
- Analyse des posts longs vs courts  

---




plt.tight_layout()
plt.savefig("images/sentiment_distribution.png")

