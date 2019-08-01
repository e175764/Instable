#https://qiita.com/shiita0903/items/838d50598cc28766f84e
import keras
import json
from keras import layers, models, optimizers
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation,regularizers
from keras.utils import np_utils
import numpy as np
import my_pickle as mp
import keras.backend as K
'''
モデル作成用のmodeling.py実行の際は、set.jsonファイルにinflation.pyで水増しをした画像の保存ディレクトリ、作成したモデルを保存するディレクトリを記述しているため、変更する必要がある。
'''
#jsonファイルからディレクトリを読み込む
f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)
sav = json_data["sav"]
model_name = json_data["model"]
weight = json_data["weight"]
#画像読み込みの際の基本情報設定
img_rows, img_cols = 32,32
img_channels = 3
#inflation.pyで.sav形式で保存した画像を読み込む。
X_train, X_test, y_train, y_test = mp.pickle_load(sav+"/sweets.sav")
'''
判別して欲しいカテゴリーは今回、イラストかそうでないか、カラフルかそうでないか、明るいかそうでないか、鮮やかかそうでないか、夜景かくらい画像かなどがはいる。
'''
categories = ["sweets映えてる","sweets映えてない"]
#categories = ["building","food","pet","view"]
#カテゴリーの数
nb_classes = len(categories)

#データの正規化
X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)
#モデル作成
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
#model.add(Dropout(0.3))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid"))#４の場合softmax,drinkでsoftmax試してみる
#モデル構成の確認
model.summary()
#モデルのコンパイル
"""
"""
opt_adam = keras.optimizers.adam(lr=0.001,decay=1e-6)
model.compile(loss="categorical_crossentropy", # 誤差(損失)関数
              optimizer=opt_adam, # 最適化関数
              metrics=["accuracy"] # 評価指標
             )

# TensorBoardで学習の進捗状況をみる
tb_cb = keras.callbacks.TensorBoard(log_dir='/Users/tuhadaiki/Instable/colorful', histogram_freq=1)
# バリデーションロスが下がれば、エポックごとにモデルを保存
cp_cb = keras.callbacks.ModelCheckpoint(filepath=weight+"/sweets.h5", monitor='val_loss',
verbose=1, save_best_only=True, mode='auto')
# バリデーションロスが3エポック連続で上がったら、ランを打ち切る
es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
verbose=0, mode='auto')
cbks = [tb_cb, cp_cb, es_cb]
#fitを行う
history = model.fit(X_train,y_train,epochs=10,
									batch_size=6,
									validation_data=(X_test,y_test)
									#,callbacks=cbks
									)
"""
グラフはaccとval_acc,lossとval_lossの折れ線グラフがそれぞれaccuracy.jpg,loss.jpgという名前で保存される。
"""
#グラフの出力           
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
#hdf5_file = weight+"/sweets.h5"
#model.save(hdf5_file)
#ベストパラメータの表示
model.load_weights(model_name+'/sweets.h5')
# テストデータに対する評価値
score = model.evaluate(X_test, y_test, verbose=0) 
print('Test score:', score[0]) # 損失関数の値 print('Test accuracy:', score[1])
print(y_test)

print(model.predict(X_test))


