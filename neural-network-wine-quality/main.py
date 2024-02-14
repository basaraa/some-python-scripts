import keras.callbacks
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD



def get_random_klasif(datas):
    total_count_train = len(datas["quality"])
    values, counts = np.unique(datas["quality"], return_counts=True)
    random_klasif = (((counts[0] / total_count_train) * (counts[0] / total_count_train)) + (
                    (counts[1] / total_count_train) * (counts[1] / total_count_train)))*100
    print ("Úspešnosť pri náhodnom klasifikátore: "+str(random_klasif))


data_vina_train = pd.read_csv("wine_train.csv")
data_vina_test = pd.read_csv("wine_test.csv")
data_vina_train.head()
data_vina_test.head()
data_vina_train.dropna(inplace=True)
data_vina_test.fillna(data_vina_train.mean(), inplace=True)
pd.options.display.max_columns = None

#povodne trenovacie

print("Trenovacie dáta pred normalizáciou:")
print (data_vina_train.describe().loc[['mean','std']])

#normalizovane trenovacie std mean
print("............................................................")
print("Trenovacie dáta po normalizácii:")
fiter=MinMaxScaler().fit(data_vina_train)
normalized_train=pd.DataFrame(fiter.transform(data_vina_train),columns=data_vina_train.columns)
print (normalized_train.describe().loc[['mean','std']])

#povodne testovacie
print("............................................................")
print("Testovacie dáta pred normalizáciou:")
print (data_vina_test.describe().loc[['mean','std']])

#normalizovane testovacie std mean
print("............................................................")
print("Testovacie dáta po normalizácii:")

normalized_test=pd.DataFrame(fiter.transform(data_vina_test),columns=data_vina_test.columns)
print (normalized_test.describe().loc[['mean','std']])

#histogram pre alkohol trénovacích dát
pl.title("Histogram alkoholu pre trénovacie dáta")
data_vina_train["alcohol"].plot(kind='hist',label="Trénovacie dáta pred normalizáciou")
normalized_train["alcohol"].plot(kind='hist',label="Trénovacie dáta po normalizácii")
pl.ylabel("Počet vzoriek")
pl.xlabel("Alkohol")
pl.legend()
pl.show()

#histogram pre alkohol testovacích dát
pl.title("Histogram alkoholu pre testovacie dáta")
data_vina_test["alcohol"].plot(kind='hist',label="Testovacie dáta pred normalizáciou")
normalized_test["alcohol"].plot(kind='hist',label="Testovacie dáta po normalizácii")
pl.ylabel("Počet vzoriek")
pl.xlabel("Alkohol")
pl.legend()
pl.show()

data_vina_train=normalized_train
data_vina_test=normalized_test

#random klasifikator
print("............................................................")
get_random_klasif(data_vina_test)
print("............................................................")

#klasifikácia cez logic regression
data_train_x=data_vina_train.drop("quality",axis=1)
data_test_x=data_vina_test.drop("quality",axis=1)
data_train_y=data_vina_train["quality"]
data_test_y=data_vina_test["quality"]
log_regresion=LogisticRegression(random_state=0,max_iter=1000).fit(data_train_x,data_train_y.values.ravel())
log_prediction=log_regresion.predict(data_test_x)
print(classification_report(data_test_y,log_prediction))
print("............................................................")

#trenovanie
data_train_x1,data_valid_x,data_train_y1,data_valid_y,=train_test_split(data_train_x,data_train_y,test_size=0.20,random_state=42)
y_train_set=to_categorical(data_train_y1)
data_valid_set=to_categorical(data_valid_y)
architektura=Sequential()
architektura.add(Dense(64,input_shape=(data_train_x.shape[1],),activation='relu'))
architektura.add(Dense(64,activation='relu'))
architektura.add(Dense(2,activation='sigmoid'))
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)
training_set=SGD(learning_rate=0.01)
architektura.compile(loss='categorical_crossentropy',optimizer=training_set,metrics=['accuracy'])
trenovanie=architektura.fit(data_train_x1,y_train_set,epochs=500,batch_size=100,verbose=0,validation_data=(data_valid_x,data_valid_set),callbacks=early_stop)

#Vývoj chyby a úspešnosti pre trénovaciu a validačnú množinu na grafe
historia_trenovania=pd.DataFrame.from_dict(trenovanie.history)
pl.title("Vývoj chyby")
historia_trenovania['loss'].plot(label="Loss")
historia_trenovania['val_loss'].plot(label="Val_Loss")
pl.ylabel("Chyba")
pl.xlabel("Epocha")
pl.legend()
pl.show()
historia_trenovania=pd.DataFrame.from_dict(trenovanie.history)
pl.title("Vývoj accuracy")
historia_trenovania['accuracy'].plot(label="Accuracy")
historia_trenovania['val_accuracy'].plot(label="Val_accuracy")
pl.ylabel("Accuracy")
pl.xlabel("Epocha")
pl.legend()
pl.show()

#Klasifikator celkova uspesnost
prediction_test=np.argmax(architektura.predict(data_test_x),axis=-1)
print(classification_report(data_test_y.values,prediction_test))
print("............................................................")
prediction_train=np.argmax(architektura.predict(data_train_x),axis=-1)
print(classification_report(data_train_y.values,prediction_train))

#Klasifikator konfuzna matica
matica_test=pd.DataFrame(confusion_matrix(data_test_y,prediction_test))
matica_train=pd.DataFrame(confusion_matrix(data_train_y,prediction_train))
pl.title("Konfuzna matica pre testovacie data")
ax=sb.heatmap(matica_test, annot=True, fmt='g')
ax.invert_yaxis()
pl.show()
pl.title("Konfuzna matica pre trenovacie data")
ax=sb.heatmap(matica_train, annot=True, fmt='g')
ax.invert_yaxis()
pl.show()

