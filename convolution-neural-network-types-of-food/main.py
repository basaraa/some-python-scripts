import os
import time
import seaborn as sb
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from keras.preprocessing.image import img_to_array,load_img,array_to_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D
from keras.regularizers import L1,L2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
import warnings
from sklearn.linear_model import SGDClassifier
from numpy import expand_dims
import random
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_train_data(cesta_train,train_datagen):
    train_data = train_datagen.flow_from_directory(
        cesta_train,
        target_size=(40, 40), batch_size=32,
        class_mode='categorical', subset='training')

    return train_data


def get_valid_data(cesta_train,train_datagen):
    valid_data = train_datagen.flow_from_directory(
        cesta_train,
        target_size=(40, 40), batch_size=32,
        class_mode='categorical', subset='validation')
    return valid_data


def get_test_data(cesta_test):
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_data = test_datagen.flow_from_directory(
        cesta_test, target_size=(40, 40), batch_size=32,
        class_mode='categorical',shuffle=False)
    return test_data


def zobraz_povodne_obrazky(cesta_train):
    graf, items = pl.subplots(5, 6, figsize=(18, 18))
    all_obrazky = sorted(os.listdir(cesta_train))
    index_obrazok = 0
    for i in range(5):
        for j in range(6):
            try:
                images_selected = all_obrazky[index_obrazok]
                index_obrazok += 1
            except:
                break
            selected_images = os.listdir(os.path.join(cesta_train, images_selected))
            nahodny_nazov = np.random.choice(selected_images)
            nahodny_obrazok = pl.imread(os.path.join(cesta_train, images_selected, nahodny_nazov))
            items[i][j].imshow(nahodny_obrazok)
            items[i][j].set_title(images_selected, fontsize=18)
            items[i][j].axis('off')
    graf.suptitle("Pôvodné obrázky z každej kategórie ", fontsize=30)
    pl.show()


def zobraz_upravene_obrazky(data_train):
    graf, items = pl.subplots(5, 6, figsize=(18, 18))
    obrazky = data_train.next()
    index_obrazok = 0
    for i in range(5):
        for j in range(6):
            items[i][j].imshow(obrazky[0][index_obrazok])
            items[i][j].axis('off')
            index_obrazok += 1
    graf.suptitle("Náhodné obrázky po načítaní a normalizácii", fontsize=30)
    pl.show()


def vyvoj_chyby_uspesnosti(trenovanie):
    historia_trenovania = pd.DataFrame.from_dict(trenovanie.history)
    pl.title("Vývoj chyby")
    historia_trenovania['loss'].plot(label="Loss")
    historia_trenovania['val_loss'].plot(label="Val_Loss")
    pl.ylabel("Chyba")
    pl.xlabel("Epocha")
    pl.legend()
    pl.show()
    pl.title("Vývoj accuracy")
    historia_trenovania['accuracy'].plot(label="Accuracy")
    historia_trenovania['val_accuracy'].plot(label="Val_accuracy")
    pl.ylabel("Accuracy")
    pl.xlabel("Epocha")
    pl.legend()
    pl.show()


def konfuzna_matica(data,prediction,triedy, nazov):
    matica = confusion_matrix(data.classes, prediction)
    pl.figure(figsize=(18,18))
    pl.title(nazov,fontsize=20)
    ax = sb.heatmap(matica, annot=True, fmt='g',annot_kws={"fontsize":19}, xticklabels=triedy, yticklabels=triedy)
    ax.invert_yaxis()
    pl.show()


def trenovanie_regularizator_L1(data_train,data_test,data_valid,train_length,test_length,valid_length,triedy):
    reguralizatory_value = [0.01, 0.001, 0.0001]
    for reguralizator in reguralizatory_value:
        time_now = time.time()
        architektura = Sequential()
        architektura.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu', padding='same'))
        architektura.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        architektura.add(MaxPooling2D(pool_size=(2, 2)))
        architektura.add(Dropout(0.2))
        architektura.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        architektura.add(GlobalAveragePooling2D())
        architektura.add(Dense(256, activation='relu'))
        architektura.add(Dropout(0.3))
        architektura.add(Dense(30, activation='softmax', kernel_regularizer=L1(reguralizator)))
        batch = 32
        optimizer = Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True,
                                         save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=12)
        architektura.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        trenovanie = architektura.fit(data_train, steps_per_epoch=train_length / batch, batch_size=batch,
                                          validation_data=data_valid, validation_steps=valid_length / batch,
                                          epochs=50, callbacks=[checkpoint, earlystopping])
        time_end = time.time()
        # Vývoj chyby a úspešnosti pre trénovaciu a validačnú množinu na grafe
        vyvoj_chyby_uspesnosti(trenovanie)
        # Celkova uspesnost
        prediction_train = np.argmax(architektura.predict(data_train, batch_size=batch, steps=train_length / batch),
                                         axis=1)
        chyba_train, uspesnost_train = architektura.evaluate(data_train, batch_size=batch, steps=train_length / batch)
        prediction_test = np.argmax(architektura.predict(data_test, batch_size=batch, steps=test_length / batch),
                                        axis=1)
        chyba_test, uspesnost_test = architektura.evaluate(data_test, batch_size=batch, steps=test_length / batch)
        print ("Regularizator: L1")
        print("Hodnota: "+str(reguralizator))
        print("Dĺžka trénovania je: " + str(time_end - time_now) + " sekúnd")
        print("Celková úspešnosť pri trénovacích dátach je: " + str(uspesnost_train * 100) + "%")
        print("Celková úspešnosť pri testovacích dátach je: " + str(uspesnost_test * 100) + "%")

        # Konfuzne matice pre trenovacie a testovacie data
        konfuzna_matica(data_train, prediction_train, triedy, "Konfuzna matica pre trenovacie data")
        konfuzna_matica(data_test, prediction_test, triedy, "Konfuzna matica pre testovacie data")


def trenovanie_regularizator_L2(data_train, data_test, data_valid, train_length, test_length, valid_length,triedy):
    reguralizatory_value = [0.01, 0.001, 0.0001]
    for reguralizator in reguralizatory_value:
        time_now = time.time()
        architektura = Sequential()
        architektura.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu', padding='same'))
        architektura.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        architektura.add(MaxPooling2D(pool_size=(2, 2)))
        architektura.add(Dropout(0.2))
        architektura.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        architektura.add(GlobalAveragePooling2D())
        architektura.add(Dense(256, activation='relu'))
        architektura.add(Dropout(0.3))
        architektura.add(Dense(30, activation='softmax', kernel_regularizer=L2(reguralizator)))
        batch = 32
        optimizer = Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True,
                                     save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=12)
        architektura.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        trenovanie = architektura.fit(data_train, steps_per_epoch=train_length / batch, batch_size=batch,
                                      validation_data=data_valid, validation_steps=valid_length / batch,
                                      epochs=50, callbacks=[checkpoint, earlystopping])
        time_end = time.time()
        # Vývoj chyby a úspešnosti pre trénovaciu a validačnú množinu na grafe
        vyvoj_chyby_uspesnosti(trenovanie)
        # Celkova uspesnost
        prediction_train = np.argmax(architektura.predict(data_train, batch_size=batch, steps=train_length / batch),
                                     axis=1)
        chyba_train, uspesnost_train = architektura.evaluate(data_train, batch_size=batch, steps=train_length / batch)
        prediction_test = np.argmax(architektura.predict(data_test, batch_size=batch, steps=test_length / batch),
                                    axis=1)
        chyba_test, uspesnost_test = architektura.evaluate(data_test, batch_size=batch, steps=test_length / batch)
        print("Regularizator: L2")
        print("Hodnota: " + str(reguralizator))
        print("Dĺžka trénovania je: " + str(time_end - time_now) + " sekúnd")
        print("Celková úspešnosť pri trénovacích dátach je: " + str(uspesnost_train * 100) + "%")
        print("Celková úspešnosť pri testovacích dátach je: " + str(uspesnost_test * 100) + "%")

        # Konfuzne matice pre trenovacie a testovacie data
        konfuzna_matica(data_train, prediction_train, triedy, "Konfuzna matica pre trenovacie data")
        konfuzna_matica(data_test, prediction_test, triedy, "Konfuzna matica pre testovacie data")


def imagenet_nacitanie(cesta,nazov):
    subory_test = []
    data=[]
    triedy=[]
    for subor in os.listdir(cesta):
        if os.path.isdir(cesta + subor):
            for podsubor in os.listdir(cesta + subor):
                if podsubor.endswith('.png') or podsubor.endswith('.jpg'):
                    subory_test.append(subor + '/' + podsubor)
    for subor in subory_test:
        obrazok = load_img(cesta + subor, target_size=(40, 40))
        x = subor.split('/')
        triedy.append(x[0])
        obrazok_value = img_to_array(obrazok)
        obrazok_value = obrazok_value / 255
        obrazok_value = preprocess_input(obrazok_value)
        data.append(obrazok_value)
    np.save(nazov+'.npy',np.array(data))
    np.save(nazov+'_triedy'+'.npy',np.array(triedy))
    print ("hotovo ulozene")


def get_image_na_augmentaciu(cesta):
    obrazok = []
    for subor in os.listdir(cesta):
        if os.path.isdir(cesta + subor):
            for podsubor in os.listdir(cesta + subor):
                if podsubor.endswith('.png') or podsubor.endswith('.jpg'):
                    obrazok.append(subor + '/' + podsubor)
    random_index = random.randrange(len(obrazok))
    img = load_img(cesta + obrazok[random_index])
    img_data = img_to_array(img)
    img_data = expand_dims(img_data, 0)
    return img_data


def vertikalny_shift_augmentacia(img):
    datagen = ImageDataGenerator(height_shift_range=0.3)
    data = datagen.flow(img,batch_size=1)
    pl.suptitle("Augmentacia obrázka veritikálnym shiftom")
    for i in range(9):
        pl.subplot(330 + 1 + i)
        batch = data.next()
        image = batch[0].astype('uint8')
        pl.imshow(image)
        pl.axis("off")
    pl.show()


def random_brightness_augmentacia(img):
    datagen = ImageDataGenerator(brightness_range=[0.4,1.0])
    data = datagen.flow(img, batch_size=1)
    pl.suptitle("Augmentacia obrázka random brightness")
    for i in range(9):
        pl.subplot(330 + 1 + i)
        batch = data.next()
        image = batch[0].astype('uint8')
        pl.imshow(image)
        pl.axis("off")
    pl.show()


def random_rotacia_augmentacia(img):
    datagen = ImageDataGenerator(rotation_range=90)
    data = datagen.flow(img, batch_size=1)
    pl.suptitle("Augmentacia obrázka random rotáciou")
    for i in range(9):
        pl.subplot(330 + 1 + i)
        batch = data.next()
        image = batch[0].astype('uint8')
        pl.imshow(image)
        pl.axis("off")
    pl.show()

cesta_train='../img/train/'
cesta_test='../img/test/'

# vytvorenie trenovacej, validacnej a testovacej mnoziny
train_datagen = ImageDataGenerator(rescale=1/255,validation_split=0.2)
data_train=get_train_data(cesta_train,train_datagen)
data_valid=get_valid_data(cesta_train,train_datagen)
data_test=get_test_data(cesta_test)
triedy=list(data_test.class_indices.keys())
train_length=len(data_train.classes)
test_length=len(data_test.classes)
valid_length=len(data_valid.classes)

# zobrazenie povodnych obrázkov z kazdej kategórie
zobraz_povodne_obrazky(cesta_train)

# zobrazenie 30 nahodnych normalizovanych obrazkov
zobraz_upravene_obrazky(data_train)

# povodne trenovanie
architektura = Sequential()
architektura.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu',padding='same'))
architektura.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
architektura.add(MaxPooling2D(pool_size=(2, 2)))
architektura.add(Dropout(0.2))
architektura.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
architektura.add(GlobalAveragePooling2D())
architektura.add(Dense(256, activation='relu'))
architektura.add(Dropout(0.3))
architektura.add(Dense(30, activation='softmax'))
batch = 32
optimizer = Adam(learning_rate=0.001)
checkpoint = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True,
                                 save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=12)
architektura.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
trenovanie = architektura.fit(data_train, steps_per_epoch=train_length / batch, batch_size=batch,
                                  validation_data=data_valid, validation_steps=valid_length / batch,
                                  epochs=50, callbacks=[checkpoint, earlystopping])

# Vývoj chyby a úspešnosti pre trénovaciu a validačnú množinu na grafe
vyvoj_chyby_uspesnosti(trenovanie)

# Celkova uspesnost
prediction_train=np.argmax(architektura.predict(data_train,batch_size=batch,steps=train_length/batch),axis=1)
chyba_train, uspesnost_train=architektura.evaluate(data_train, batch_size=batch, steps=train_length / batch)
prediction_test=np.argmax(architektura.predict(data_test,batch_size=batch,steps=test_length/batch),axis=1)
chyba_test, uspesnost_test=architektura.evaluate(data_test, batch_size=batch, steps=test_length / batch)
print("Celková úspešnosť pri trénovacích dátach je: " + str(uspesnost_train * 100) + "%")
print ("Celková úspešnosť pri testovacích dátach je: "+str(uspesnost_test*100)+"%")
# Konfuzne matice pre trenovacie a testovacie data
konfuzna_matica(data_train,prediction_train,triedy,"Konfuzna matica pre trenovacie data")
konfuzna_matica(data_test,prediction_test,triedy,"Konfuzna matica pre testovacie data")

# pouzitie L1 reguralizatora
trenovanie_regularizator_L1(data_train,data_test,data_valid,train_length,test_length,valid_length,triedy)

# pouzitie L2 reguralizatora
trenovanie_regularizator_L2(data_train,data_test,data_valid,train_length,test_length,valid_length,triedy)

# nacitanie datasetu
imagenet_nacitanie(cesta_test,"imagenet_test")
imagenet_nacitanie(cesta_train,"imagenet_train")
data_test=np.load("imagenet_test.npy")
triedy_test=np.load("imagenet_test_triedy.npy")
data_train=np.load("imagenet_train.npy")
triedy_train=np.load("imagenet_train_triedy.npy")

# zakodovanie datasetu
res_model=ResNet50(weights='imagenet',input_shape=(40,40,3),include_top=False)
predicted_test=res_model.predict(data_test)
predicted_train=res_model.predict(data_train)
np.save("imagenet_test_predicted.npy",np.array(predicted_test))
np.save("imagenet_train_predicted.npy",np.array(predicted_train))
predicted_test=np.load("imagenet_test_predicted.npy")
predicted_train=np.load("imagenet_train_predicted.npy")

# redukcia do 2D test
pca_test = PCA(n_components=2)
samples,b,c,res_triedy = predicted_test.shape
reshape_test = predicted_test.reshape((samples,b*c*res_triedy))
pca_test.fit(reshape_test)
redukovane_test = pca_test.transform(reshape_test)

# redukcia do 2D train
pca_train = PCA(n_components=2)
samples,b,c,res_triedy = predicted_train.shape
reshape_train = predicted_train.reshape((samples,b*c*res_triedy))
pca_train.fit(reshape_train)
redukovane_train = pca_train.transform(reshape_train)

# plotovanie priznakov test
redukovane_test1 = redukovane_test[:, 0]
redukovane_test2 = redukovane_test[:, 1]
pl.subplots(figsize=(15, 15))
for trieda in triedy:
    pl.scatter(redukovane_test1[triedy_test == trieda][:], redukovane_test2[triedy_test == trieda][:], alpha=0.4, label=trieda)
pl.legend()
pl.title("Vizualizácia príznakov v 2D na testovacej množine")
pl.show()

# plotovanie priznakov train
redukovane_train1 = redukovane_train[:, 0]
redukovane_train2 = redukovane_train[:, 1]
pl.subplots(figsize=(15, 15))
for trieda in triedy:
    pl.scatter(redukovane_train1[triedy_train == trieda][:], redukovane_train2[triedy_train == trieda][:], alpha=0.4, label=trieda)
pl.legend()
pl.title("Vizualizácia príznakov v 2D na trénovacej množine")
pl.show()

# nenahodny klasifikator SGD
clf = SGDClassifier(max_iter=500,early_stopping=True,verbose=1,validation_fraction=0.2)
clf.fit(reshape_train, triedy_train)
print ("Úspešnosť na trénovacích dátach ResNet50 SGD: "+str(clf.score(reshape_train,triedy_train)*100))
print ("Úspešnosť na testovacích dátach ResNet50 SGD: "+str(clf.score(reshape_test,triedy_test)*100))

# Bonus augmentation vertical shift, random brightness a random rotation
img_aug=get_image_na_augmentaciu(cesta_test)
vertikalny_shift_augmentacia(img_aug)
random_brightness_augmentacia(img_aug)
random_rotacia_augmentacia(img_aug)

