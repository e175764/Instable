import keras
import json
from keras import layers, models, optimizers
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation,regularizers
from keras.utils import np_utils
import numpy as np
'''
モデル作成用のmodeling.py実行の際は、set.jsonファイルにinflation.pyで水増しをした画像の保存ディレクトリ、作成したモデルを保存するディレクトリを記述しているため、変更する必要がある。
'''
f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)
npy = json_data["sav"]
model_name = json_data["model"]
weight = json_data["weight"]
img_rows, img_cols = 32,32
img_channels = 3
X_train, X_test, y_train, y_test = np.load(npy + "/food_circle.sav")
#データの処理
'''
判別して欲しいカテゴリーは今回、イラストかそうでないか、カラフルかそうでないか、明るいかそうでないか、鮮やかかそうでないか、夜景かくらい画像かなどがはいる。
'''
categories = ["food_circle映えてない","food_circle映えてる"]
nb_classes = len(categories)

#データの正規化
X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


"""
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(256,activation="relu"))

model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定
"""


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
model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定softmaxだと0%,99%など値が偏る。

#モデル構成の確認
model.summary()
#モデルのコンパイル変える

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["accuracy"])        
              
model.fit(X_train,
	y_train,
	epochs=10,
	batch_size=6,
	validation_data=(X_test,y_test))
          
json_string = model.model.to_json()
open(model_name +'/food_circle.json', 'w').write(json_string)

#重みの保存

hdf5_file = weight+"/food_circle.hdf5"
model.model.save_weights(hdf5_file)
        
import matplotlib.pyplot as plt
"""
グラフはaccとval_acc,lossとval_lossの折れ線グラフがそれぞれaccuracy.jpg,loss.jpgという名前で保存される。
"""
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

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
