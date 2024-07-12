# Description du repo

- lstm_code : code python pour l'entrainement et la prédiction d'un modèle LSTM + RESNET pour la prédiction de la RPE à partir de la puissance des séances.
- notebooks : contient pas mal de notebooks de tests majoritairement. Les principaux sont clean_prod.ipynb, script_plot.ipynb et vis_pres.ipynb
- utils : continet des fonctions utiles pour le traitement des données et la création des modèles. Si d'autres fonctions ne fonctionne pas, c'est probablement du à des déplacer des fichiers. Notamment la normalisation et le formattage des données.
- script_hmm : contient des codes pour l'entrainement et la prédiction d'un modèle HMM pour la détéction de zones de puissances. hmm_script.py est le script principal pour l'entrainement et la prédiction. hmm_plot.py est un script pour la visualisation des résultats. Des exemples d'exécutions 
- test : contient des fichiers de tests pour les fonctions du dossier utils.
- pres : contient des fichiers pour la présentation du projet, des slides et des figures.

# TODO :
- Faire la pipeline de puissance par état.
- check qu'il n'y a pas d'info perso dans les notebooks ? 

# Formattage des données pour l'entrainement des modèles et nom de columne

- Cleaned columns : 'id_session','tps','stream_watts','stream_heartrate','date','ath_id'
- Meta_data columns : 'id_session','poids','date','rpe', 'sport','ath_id','id' (qui est ath_id*1000000+id_session)
- normalized columns : 'id_session','tps','stream_watts','stream_heartrate','date','rpe','sport','ath_id','id'
- norm_data : 'id_session','ppr', 'ma_hr','roll_std_hr','rpe','date','sport'
- Sport : ['Vélo - Route','Vélo - Home Trainer', 'Vélo - Piste','Vélo - CLM']