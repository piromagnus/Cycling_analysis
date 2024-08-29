# TODO :
- Faire la pipeline de puissance par état.
- check qu'il n'y a pas d'info perso dans les notebooks ? 


# Idée du stage
  Utiliser les données d'entraînements des athlètes pour définir un niveau de forme au court du temps. L'idée est de montrer et de quantifier la progression d'une séance à l'autre.
  L'approche est de prédire la RPE (Rate of Perceived Exertion) à partir de la puissance des séances. La RPE est une mesure subjective de l'intensité de l'effort. Elle est utilisée pour quantifier l'intensité de l'effort fourni par un athlète. Le modèle d'apprentissage profond utilisé est un LSTM + RESNET pour prédire la RPE à partir de la puissance des séances en utilisant l'espace latent comme représentation de la forme de l'athlète.

# Fonctionnement
Deux étapes principales sont nécessaires pour parvenir à cela.
1. Transformation des données brutes par secondes de chaque séance en données normalisées par séance pour avoir un suivi longitutudinal. L'approche utilisé ici est de définir des zones de puissance avec des modèles de markov cachés et de calculer le temps passer dans chaque zones de puissances par séance.
2. Entrainement d'un modèle de machine learning recursif avec un espace latent qui représente la forme de l'athlète. Le modèle est un LSTM + RESNET qui prend en entrée les données normalisées (de puissance) par séance et prédit la RPE.

# Description du repo

- lstm_code : code python pour l'entrainement et la prédiction d'un modèle LSTM + RESNET pour la prédiction de la RPE à partir de la puissance des séances.
- notebooks : contient pas mal de notebooks de tests majoritairement. Les principaux sont clean_prod.ipynb, script_plot.ipynb et vis_pres.ipynb
- utils : continet des fonctions utiles pour le traitement des données et la création des modèles. Si d'autres fonctions ne fonctionne pas, c'est probablement du à des déplacer des fichiers. Notamment la normalisation et le formattage des données.
- script_hmm : contient des codes pour l'entrainement et la prédiction d'un modèle HMM pour la détéction de zones de puissances. hmm_script.py est le script principal pour l'entrainement et la prédiction. hmm_plot.py est un script pour la visualisation des résultats. Des exemples d'exécutions 
- test : contient des fichiers de tests pour les fonctions du dossier utils.
- pres : contient des fichiers pour la présentation du projet, des slides et des figures.


# Formattage des données pour l'entrainement des modèles et nom de columne

- Cleaned columns : 'id_session','tps','stream_watts','stream_heartrate','date','ath_id'
- Meta_data columns : 'id_session','poids','date','rpe', 'sport','ath_id','id' (qui est ath_id*1000000+id_session)
- normalized columns : 'id_session','tps','stream_watts','stream_heartrate','date','rpe','sport','ath_id','id'
- norm_data : 'id_session','ppr', 'ma_hr','roll_std_hr','rpe','date','sport'
- Sport : ['Vélo - Route','Vélo - Home Trainer', 'Vélo - Piste','Vélo - CLM']


# Étape du stage
1. Bibliographie en science des sports (modèles de puissance critique, loi de puissance, prediction de blessures, modèles statistiques, filières énergetiques...)
2. Bibliographie en macchine learning (LSTM, RESNET, HMM, Transformers, SSM ...)
3. exploration des données et prétraitement
4. Réflexion sur l'efficacité musculaire (ratio energie utile pour le mouvement / energie dépensée par le corps) avec l'objectif de le quantifié au cours de l'effort.
5. Essai d'utilisation du Profil de puissance par séance pour comparer les séances mais ça décrit uniquement les puissances max et on a pas d'info sur le nombre de fois effectués. (dans le cas d'un fractionné par exemple).
6. Zones de puissances à partir du PPR et des temps classiques pour chaque type d'effort (endurance, seuil, sprint) mais fonctionne assez mal car c'est seuil temporel sont très variables d'un athlète à l'autre.
7. Utilisation de HMM pour définir les zones de puissances et calculer le temps passé dans chaque zone par séance. Utilisation aussi sur la puissance cardio pour estimer la puissance moyenne dans chaque zone de puissance cardiaque par séance et son evolution.
8. Entraînement d'un LSTM + RESNET pour prédire la RPE à partir de la puissance normalisée par séance. Beaucoup de temps pour optimiser les hyper paramètres définir les fonctions de coûts etc. très bon sur les données d'entrainement mais pas sur les données de test.Overfitting probablement mais peut-être que le raisonnement n'est pas bon car on charge à la fois à optimiser le réseau et aussi à trouver l'état caché qui représente la forme de l'athlète sans superviser autrement qu'avec la RPE de la séance. Possibilité d'un manque d'information sur les conditions de l'entraînement. 
9. Le modèle devait aussi servir pour déterminer les entraînements particulièrement mauvais (car il ne prend pas encore le sommeil, la récupération etc.) mais comment savoir si c'est le modèle qui est mauvais ou si l'athlète est hors de ses niveaux de perfs ? 
