# Projet Spark : Campagne Kickstarter

*Thomas Mensch* [thomas.mensch@gmail.com](mailto:thomas.mensch@gmail.com)

### Organisation du répertoire

Ce répertoire les fichiers et répertoires suivants:

 - `README.md`: ce fichier
 - `build_and_submit.sh`: script bash pour éxecuter les scripts *Scala*
 - `src/main/scala/paristech`: répertoire contenant le code source en *Scala*
 - `resources/train`: répertoire contenant les données initiales au format `.csv` 
 
### Traitement des données

Le traitement des données s'effectue en deux étapes:
 1. Préparation du jeu de données
 
 	`./build_and_submit.sh Preprocessor`
 
 2. Construction d'un classifieur
 	`./build_and_submit.sh Trainer` 

#### Préparation du jeu de données

Le script `Preprocessor` prépare

Les données sont sauvegardées au format *parquet*

#### Construction d'un classifieur

#### Résultats obtenus et tentative d'amélioration

Les résultats obtenus sont l
Before grid search we have a f1-score of 0.613

After grid-search the f1-score increases to 0.652