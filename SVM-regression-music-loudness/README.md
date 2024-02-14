# Projekt SVM regresie - predpovedanie hlasitosti hudby
Projekt vznikol v rámci školského zadania v roku 2021. Projekt je zameraný na predpovedanie hlasitosti hudby pomocou SVM regresie. Projekt mal na vstupe veľké množstvo dát so 17 parametrami o hudbe zo spotify na základe ktorých predpovedalo hlasitosť hudby.
Zoznam funkcionalít:
-  načítanie a normalizovanie dát, analyzovanie potrebných a nepotrebných parametrov
-  odstránenie nepotrebných parametrov a tiež hraničných hodnôt a duplikátov 
-  zakódovanie string hodnôt
-  natrénovanie SVM regresora na predpoveď hlasitosti piesne s použitím krížovej validácie
-  klasifikovanie cez MSE a R^2 metriku
-  použitie mriežkového vyhľadávania
-  použitie súborového učenia s použitím bagging a boosting metódou učenia (adaboost a randomforest)

V projekte sú použité nasledujúce python knižnice:
-  pandas
-  ast
-  scripy
-  collection
-  seaborn
-  pyplot
-  sklearn
-  numpy

# Project SVM regression - predicting loudness of music
Project has been created as school assignment in 2021. This project was focused on predicting loudness of music by SVM regression. Project had plenty of test and train datas with 17 parameters as input information about music and then it was predicting loudness of music.
List of features:
-  loading and normalization of input datas, analysis useful and useless parameters
-  delete useless parameters, also threshold and duplicate values
-  classification with MSE and R^2 metrics
-  training SVM regressor for predicting loudness of music with cross validation
-  using grid search
-  using ensemble learning with bagging and boosting learning methods (adaboost and randomforest)

In this project are used these python libraries:
-  pandas
-  ast
-  scripy
-  collection
-  seaborn
-  pyplot
-  sklearn
-  numpy
