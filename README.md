# Projet Spark : Campagnes Kickstarter

*Thomas Mensch* [thomas.mensch@gmail.com](mailto:thomas.mensch@gmail.com)

Le projet se trouve sur github: 

[https://github.com/ThomasMensch/spark_project_kickstarter_2019_2020_TMensch](https://github.com/ThomasMensch/spark_project_kickstarter_2019_2020_TMensch)

### Introduction

 - **Objectif**: Le but de ce projet est de construire un modèle/classifieur capable de prédire si une campagne *Kickstarter* va réussir ou non.

 - **Données**: Les données sont disponibles sur [kaggle](https://www.kaggle.com/codename007/funding-successful-projects).

  - **Au final**: nous obtenons, après une *Grid Search* et manipulation additionnelle d'une des variables explicatives (*feature engineering*), une F-mesure (ou *F1-score*) de: **0.672**. 

### Organisation du répertoire 

Ce répertoire contient les fichiers et répertoires suivants:

 - `README.md`: ce fichier
 - `build_and_submit.sh`: script bash pour éxecuter les scripts *Scala*
 - `src/main/scala/paristech`: répertoire contenant le code source en *Scala*, notamment les scripts `Preprocessor.scala` et `Trainer.scala`
 - `resources/train`: répertoire contenant les données initiales au format `.csv` 

Lors de l'éxecution des scripts *Scala*, les fichiers et répertoires suivants sont créés:

 - `resources/preprocessed`: répertoire contenant le DataFrame prétraité au format *parquet* après éxecution du script `Preprocessor.scala`.
 - `resources/best-logistic-regression-model`: répertoire contenant la sauvegarde du modèle après *"Grid Search"* ' (après éxecution du script `Trainer.scala`).
 - `resources/best-logistic-regression-model-2`: répertoire contenant la sauvegarde du modèle après *"Feature Engineering"* et *"Grid Search"* (après éxecution du script `Trainer.scala`).
 - `output.txt`: fichier, généré par le script `Trainer.scala`, résumant les différentes étapes (utilisé pour *"debug"*).

### Traitement des données

Le traitement des données s'effectue en deux étapes:

    1. Préparation du jeu de données

```shell
    ./build_and_submit.sh Preprocessor
```

    2. Construction d'un classifieur

```shell
 	./build_and_submit.sh Trainer
```

#### Préparation du jeu de données

Le script `Preprocessor` charge le jeu de données au format `.csv` et effectue le pré-traitement des données, à savoir:

 - supression des colonnes inutiles
 - nettoyage des données (supression des duplicats, des valeurs nulles ...)
 - *recasting* des colonnes en un type exploitable par l'algorithme
 - vérification de la cohérence des colonnes (mêmes unités, ...)
 - suppression des colonnes liée aux fuites du futur
 - ...

Les données pré-traitées sont sauvegardées au format *parquet*, dans le répertoire `resources\preprocessed`.
La taille du jeu de données après pré-traitement est de: *108129 lignes et 11 colonnes*.

#### Construction d'un classifieur

Pour la construction du modèle (script `Trainer.scala`, nous partons du jeu de données pré-traitée sur lequel nous effectuons un nettoyage supplémentaire supprimant 514 lignes  (suppression des valeurs mises à -1 et *Unknown* pour les variables `days_campaign`, `hour_prepa`, `goal` `country2` et `currency2`).

La taille du jeu de données pour la construction du modèle est de: *107615 lignes et 11 colonnes*.

Nous construisons ensuite le *Pipeline*, i.e., la séquence de prétraitement permettant de transformer les données dans un format utilisable par l'algorithme de classification de la bibliothèque `spark.ml`. Ici nous utilisons comme classifieur une régression logistique. 
Les principales étapes (*stages*) du *Pipeline* sont:

 - Traitement des données textuelles
    - Stage 1 : récupérer les mots des textes
    - Stage 2 : retirer les stop words
    - Stage 3 : computer la partie TF
    - Stage 4 : computer la partie IDF
 - Conversion des variables catégorielles en variables numériques
    - Stage 5 : convertir country2 en quantités numériques
    - Stage 6 : convertir currency2 en quantités numériques
    - Stages 7 et 8: One-Hot encoder ces deux catégories
 - Mettre les données sous une forme utilisable par `spark.ml`
    - Stage 9 : assembler tous les features en un unique vecteur
    - Stage 10 : créer/instancier le modèle de classification

A partir de ce *Pipeline*, nous entrainons le modèle sur un échantillon d'entrainement (90% du jeu de données).
Nous testons ce modèle sur le jeu de données de test (10% du jeu de données initial), nous obtenons les prédictions et *F1-score* suivants:

```shell
+------------+-----------+-----+
|final_status|predictions|count|
+------------+-----------+-----+
|           1|        0.0| 1844|
|           0|        1.0| 2390|
|           1|        1.0| 1624|
|           0|        0.0| 4897|
+------------+-----------+-----+

F1-score on test set [before grid search]: 0.613
```

Le *F1-score* pour ce classifieur est **0.613**.

La qualité du classifieur est mesurée à l'aide de la F-mesure (ou *F1-score*).
Cette mesure est la moyenne harmonique de la précision et du rappel (*recall*) comprise entre 0 et 1.
Le *F1-score* pour un classifieur parfait est 1.


#### Résultats obtenus et tentative d'amélioration

Afin d'ameliorer les performance de notre modèle, nous effectuons une *"Grid Search"* pour ajuster les hyper-paramètres du modèle.
Après *"Grid Search"*, nous obtenons un *"best model"* qui donne les prédictions suivantes sur l'échantillon de test:

```shell
+------------+-----------+-----+
|final_status|predictions|count|
+------------+-----------+-----+
|           1|        0.0| 1029|
|           0|        1.0| 2837|
|           1|        1.0| 2439|
|           0|        0.0| 4450|
+------------+-----------+-----+

F1-score on test set [after grid search]: 0.652
```

Le *F1-score* pour ce classifieur est **0.652**, résultat légèrement meilleur que le précédent.
Ce modèle est sauvegardé dans le répertoire `resources/best-logistic-regression-model`.

#### Pour aller un peu plus loin ... *Feature engineering*

La variable `goal` est une variable important du problème. En étudiant sa distribution, on remarque que cette dernière est très asymétrique. 
Nous choisissons de créer une nouvelle variable `goal2` correspondant au logarithme naturel de la variable `goal`.
Après modification du *Pipeline* afin de prendre en compte cette nouvelle variable et application d'une *"Grid Search"*, nous obtenons un nouveau *best model*,
dont les performances sont les suivantes:


```shell
+------------+-----------+-----+
|final_status|predictions|count|
+------------+-----------+-----+
|           1|        0.0|  946|
|           0|        1.0| 2697|
|           1|        1.0| 2522|
|           0|        0.0| 4590|
+------------+-----------+-----+

F1-score on test set [after grid search and variable engineering]: 0.672
```

Le *F1-score* pour ce classifieur est **0.672**, résultat légèrement que le précédent.
Ce modèle est sauvegardé dans le répertoire `resources/best-logistic-regression-model-2`.


D'autre part, j'ai essayé d'appliquer l'algorithme de "fôrets aléatoires", réputé plus performant sur ce jeu de données.
Malheureusement les limitations en RAM de mon ordinateur portable ne m'ont pas permis d'arriver au bout du calcul.
