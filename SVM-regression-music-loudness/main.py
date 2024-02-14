import ast
import pandas as pd
from sklearn import preprocessing,svm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
import matplotlib.pyplot as pl
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
from scipy import stats
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
pd.options.mode.chained_assignment = None


def trening_SVR(spotify_train_x1,spotify_train_y1):
    arch = svm.SVR(kernel='linear', gamma='auto', C=10)
    arch.fit(spotify_train_x1, spotify_train_y1)
    print(cross_val_score(arch, spotify_train_x1, spotify_train_y1, cv=5))
    print("MSE train: %.4f" % mean_squared_error(spotify_train_y1, arch.predict(spotify_train_x1)))
    print("R2 train: %.4f" % r2_score(spotify_train_y1, arch.predict(spotify_train_x1)))
    print("MSE test: %.4f" % mean_squared_error(spotify_test_y, arch.predict(spotify_test_x)))
    print("R2 test: %.4f" % r2_score(spotify_test_y, arch.predict(spotify_test_x)))
    y_predict = arch.predict(spotify_test_x)
    residuals = spotify_test_y - y_predict
    pl.scatter(residuals, y_predict)
    pl.ylabel("Hlasitosť")
    pl.xlabel("Reziduály")
    pl.show()


def trening_GridSearch(spotify_train_x1,spotify_train_y1):
    parametre = [
        {"kernel": ["rbf"], "gamma": ["auto", "scale"], "C": [2, 10, 112, 1002]},
        {"kernel": ["linear"], "gamma": ["scale", "auto"], "C": [2, 10, 112, 1002]},
    ]
    grid = GridSearchCV(svm.SVR(), parametre)
    grid.fit(spotify_train_x1, spotify_train_y1)
    print("Najlepšie parametre: ", grid.best_params_)
    y, y_predict = spotify_train_y1, grid.predict(spotify_train_x1)
    print("Grid MSE train: %.4f" % mean_squared_error(y, y_predict))
    print("Grid R2 train: %.4f" % r2_score(y, y_predict))
    y, y_predict = spotify_test_y, grid.predict(spotify_test_x)
    print("Grid MSE test: %.4f" % mean_squared_error(y, y_predict))
    print("Grid R2 test: %.4f" % r2_score(y, y_predict))


def trening_RandomForest(spotify_train_x1,spotify_train_y1):
    arch_forest = RandomForestRegressor()
    arch_forest.fit(spotify_train_x1, spotify_train_y1)
    print("RandomForest MSE train: %.4f" % mean_squared_error(spotify_train_y1, arch_forest.predict(spotify_train_x1)))
    print("RandomForest R2 train: %.4f" % r2_score(spotify_train_y1, arch_forest.predict(spotify_train_x1)))
    print("RandomForest MSE test: %.4f" % mean_squared_error(spotify_test_y, arch_forest.predict(spotify_test_x)))
    print("RandomForest R2 test: %.4f" % r2_score(spotify_test_y, arch_forest.predict(spotify_test_x)))


def trening_AdaBoost(spotify_train_x1,spotify_train_y1):
    arch_ada = AdaBoostRegressor(n_estimators=30)
    arch_ada.fit(spotify_train_x1, spotify_train_y1)
    print("AdaBoost MSE train: %.4f" % mean_squared_error(spotify_train_y1, arch_ada.predict(spotify_train_x1)))
    print("AdaBoost R2 train: %.4f" % r2_score(spotify_train_y1, arch_ada.predict(spotify_train_x1)))
    print("AdaBoost MSE test: %.4f" % mean_squared_error(spotify_test_y, arch_ada.predict(spotify_test_x)))
    print("AdaBoost R2 test: %.4f" % r2_score(spotify_test_y, arch_ada.predict(spotify_test_x)))


# nacitanie vstupnych dat
spotify_train = pd.read_csv("spotify_train.csv")
spotify_test = pd.read_csv("spotify_test.csv")

train_length=len(spotify_train)
# delete duplicates
spotify_train.drop_duplicates(subset=['artist_id','name'],ignore_index=True,inplace=True)
print ("Počet vymazaných duplikátov: ",(train_length-len(spotify_train)))
# delete columns
spotify_test_x=spotify_test.drop(["id","name","artist_id","query","url","duration_ms",
                                  "playlist_id","playlist_description","playlist_name",
                                  "artist_followers","popularity","playlist_url"],axis=1)
spotify_train_x=spotify_train.drop(["id","name","artist_id","query","url","duration_ms",
                                    "playlist_id","playlist_description","playlist_name",
                                    "artist_followers","popularity","playlist_url"],axis=1)
# check nan values
print ("NaN values in train:\n",spotify_train_x.isna().sum())
print ("NaN values in test:\n",spotify_train_x.isna().sum())

# enkodovanie roku
spotify_test_x['release_date'] = pd.DatetimeIndex(spotify_test_x['release_date']).year
spotify_train_x['release_date'] = pd.DatetimeIndex(spotify_train_x['release_date']).year

# enkodovanie ostatnych string hodnot
encoder = preprocessing.LabelEncoder()
spotify_test_x['explicit']=encoder.fit(spotify_test_x['explicit']).transform(spotify_test_x['explicit'])
spotify_train_x['explicit']=encoder.fit(spotify_train_x['explicit']).transform(spotify_train_x['explicit'])
spotify_test_x['artist']=encoder.fit(spotify_test_x['artist']).transform(spotify_test_x['artist'])
spotify_train_x['artist']=encoder.fit(spotify_train_x['artist']).transform(spotify_train_x['artist'])

# vymazanie artist_genres ktore sa neskor enkoduju
artist_genres_train=spotify_train_x['artist_genres']
artist_genres_test=spotify_test_x['artist_genres']
spotify_train_x.drop(["artist_genres"],inplace=True,axis=1)
spotify_test_x.drop(["artist_genres"],inplace=True,axis=1)

# korelacna matica
pl.figure(figsize=(16, 10))
sb.set(style="whitegrid")
corr = spotify_train_x.corr()
sb.heatmap(corr,annot=True,cmap="coolwarm")
pl.show()

spotify_train_x.drop(["key","mode","artist"],inplace=True,axis=1)
spotify_test_x.drop(["key","mode","artist"],inplace=True,axis=1)

train_length=len(spotify_train_x)
# delete outliers hodnot
z_score = np.abs(stats.zscore(spotify_train_x))
outliers=np.where(z_score > 3)
spotify_train_x.drop(outliers[0],inplace=True)
print ("Počet vymazaných riadkov cez hľadanie hraničných hodnôt: ",(train_length-len(spotify_train_x)))

# enkodovanie zanrov

music_genres=list()
best_genres=list()
for i in artist_genres_train:
    genre_array=ast.literal_eval(i)
    for j in genre_array:
        music_genres.append(j)

for i in artist_genres_test:
    genre_array=ast.literal_eval(i)
    for j in genre_array:
        music_genres.append(j)

for i in Counter(music_genres).most_common(8):
    best_genres.append(i[0])

for i in best_genres:
    spotify_train_x[i]=0
    spotify_test_x[i] = 0
spotify_train_x['others']=0
spotify_test_x['others']=0

for i in range (len(artist_genres_train)):
    genre_array=ast.literal_eval(artist_genres_train[i])
    for j in genre_array:
        if j in best_genres:
            spotify_train_x[j][i]=1
        else:
            spotify_train_x['others'][i] = 1

for i in range (len(artist_genres_test)):
    genre_array=ast.literal_eval(artist_genres_test[i])
    for j in genre_array:
        if j in best_genres:
            spotify_test_x[j][i]=1
        else:
            spotify_test_x['others'][i] = 1
# korelacna matica
pl.figure(figsize=(16, 10))
sb.set(style="whitegrid")
corr = spotify_train_x.corr()
sb.heatmap(corr,annot=True,cmap="coolwarm")
pl.show()

# split
spotify_train_y=spotify_train_x["loudness"]
spotify_test_y=spotify_test_x["loudness"]
spotify_train_x=spotify_train_x.drop("loudness",axis=1)
spotify_test_x=spotify_test_x.drop("loudness",axis=1)

# normalizacia
fiter=MinMaxScaler().fit(spotify_train_x)
spotify_train_x=pd.DataFrame(fiter.transform(spotify_train_x),columns=spotify_train_x.columns)
spotify_test_x=pd.DataFrame(fiter.transform(spotify_test_x),columns=spotify_test_x.columns)
spotify_train_x1=spotify_train_x
spotify_train_y1=spotify_train_y

# trenovanie pri SVR
trening_SVR(spotify_train_x1,spotify_train_y1)

# GridSearch
trening_GridSearch(spotify_train_x1,spotify_train_y1)

# RandomForest
trening_RandomForest(spotify_train_x1,spotify_train_y1)

# AdaBoost
trening_AdaBoost(spotify_train_x1,spotify_train_y1)

