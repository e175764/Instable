import json
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

f = open("set.json", 'r')
json_data = json.load(f) #JSON形式で読み込む
modelpass = json_data["model"]

#保存したモデルの読み込み
#model = model_from_json(open(modelpass +'/food_circle.json').read())
model = model_from_json(open(modelpass +'/test_sweets.json').read())
#保存した重みの読み込み
#model.load_weights(modelpass +'/food_circle.hdf5')
model.load_weights(modelpass +'/test_sweets.hdf5')

categories = ["food_circle映え","food_circle映えてない"]
#categories = ["sweets映えてる","sweets映えてない"]
#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#予測
preds = model.predict(x)
#予測結果によって処理を分ける
if preds[1]>=preds[0]:#ここの比較は個人のmodeling.pyのカテゴリー順によるので、ばらけそう。
		print("food_circle映えてない")
		#print("sweets映えてない")
else:
	print("food_circle映えてる")
	#print("sweets映え")
print(preds)
print('preds.shape: {}'.format(preds.shape))
"""
result = decode_predictions(preds, top=2)[0]
print(result)
"""