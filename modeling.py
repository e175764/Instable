#https://qiita.com/shiita0903/items/838d50598cc28766f84e
import keras
from keras import layers, models, optimizers
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation,regularizers
from keras.utils import np_utils
import numpy as np
import my_pickle as mp
import keras.backend as K

img_rows, img_cols = 32,32
img_channels = 3
X_train, X_test, y_train, y_test = mp.pickle_load("/Users/e175764/Desktop/Third/DataMining/Sea/insta_data3.sav")
#データの処理
categories = ["映える","映えない"]
nb_classes = len(categories)

#データの正規化
X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.4))

model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定

#モデル構成の確認
model.summary()
#モデルのコンパイル
             
opt_adam = keras.optimizers.adam(lr=0.001,decay=1e-6)
model.compile(loss="categorical_crossentropy", # 誤差(損失)関数
              optimizer=opt_adam, # 最適化関数
              metrics=["accuracy"] # 評価指標
             )

history = model.fit(X_train,y_train,epochs=1,batch_size=6,validation_data=(X_test,y_test))
                  
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy.jpg')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.jpg')

#モデル自身の保存
hdf5_file = "/Users/e175764/Desktop/Third/DataMining/Sea/insta2.h5"
model.save(hdf5_file)

print(y_test)

print(model.predict(X_test))
