from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import json
import os, glob
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img
import random, math
from PIL import Image
import my_pickle as mp


f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)
#画像が保存されているルートディレクトリのパス(水増し前)
root_dir = json_data["images"]
save = json_data["save_npy"]
infx = json_data["infx"]
infy = json_data["infy"]
# カテゴリ
categories = ["映える","映えない"]


#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files,flag):
    #local X, Y
    X = []
    Y = []
    for cat, fname in files:
        X,Y = add_sample(cat, fname,X,Y,flag)
    return np.array(X), np.array(Y)

#画像水増しの際に用いる関数:どういった処理を行なって増やすかを決める.
# ImageDataGeneratorを定義
datagen = ImageDataGenerator(rotation_range=30,
                            width_shift_range=20,
                            height_shift_range=0.,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True)
                            
#渡された画像データを読み込んでXに格納する
#画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname, X, Y, flag):
	img = Image.open(fname)
	img = img.convert("RGB")
	img = img.resize((150, 150))
	data = np.asarray(img)
	if(flag):
		# 1つの入力画像から何枚拡張するかを指定（今回は50枚）
		for i in range(50):
			data2 = draw_images(datagen,data)
			X.append(data2)
			Y.append(cat)
	else:
		X.append(data)
		Y.append(cat)
	return X,Y
#全データ格納用配列
allfiles=[]
big=[]
train=[]
test=[]

#カテゴリ配列の各値と、それに対応するidxを認識しtestとtrainそれぞれで読みこむ
for idx, cat in enumerate(categories):
	temp=[]
	image_dir = root_dir + cat
	files = glob.glob(image_dir + "/*.jpg")
	for f in files:
		allfiles.append((idx, f))
	random.shuffle(allfiles)
	#train:test = 8:2
	th = math.floor(len(allfiles) * 0.8)
	train += allfiles[0:th]
	test  += allfiles[th:]
	allfiles=[]
    

#実際に水増しを行う関数
def draw_images(generator,x):
	x2=(x[np.newaxis,:,:,:])
	g = generator.flow(x2,batch_size=1)
	bach=np.reshape(g.next(),(150,150,3))
	return bach

#x,yのデータがすでに存在する場合には読み込み,ない場合には新規作成してそれを読み込む
if not(os.path.exists(infy) and os.path.exists(infx)):
	X_train, y_train = make_sample(train,True)
	mp.pickle_dump(X_train,infx)
	mp.pickle_dump(y_train,infy)

X_train = mp.pickle_load(infx)
y_train = mp.pickle_load(infy)

X_test, y_test = make_sample(test,False) #Falseでは水増しなし

xy = (X_train, X_test, y_train, y_test)
#データを保存する（データの名前を「tea_data.npy」としている）
mp.pickle_dump(xy, save+"insta_data3.sav") 