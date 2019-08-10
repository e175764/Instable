from keras import models
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import json
import sys

f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)

night = json_data["night"]
illust= json_data["illust"]
colorful = json_data["colorful"]
food_cat = json_data["food_cat"]
one_food = json_data["one_food"]
course_foods = json_data["course_foods"]
sweets = json_data["sweets"]
drink = json_data["drink"]
bright = json_data["bright"]
building = json_data["building"]
photo_cat = json_data["photo_cat"]

print("モデル読み込みを開始します")
print("少々お待ちください")
#保存したモデルの読み込み
model_night=load_model(night)
model_illust=load_model(illust)
model_colorful=load_model(colorful)
model_one_food=load_model(one_food)
model_course_foods=load_model(course_foods)
print("...")
model_sweets=load_model(sweets)
model_drink=load_model(drink)
model_bright=load_model(bright)
model_building=load_model(building)
model_food_cat=load_model(food_cat)
model_categories=load_model(photo_cat)

print("モデル読み込みが終わりました")
print("判定したい画像のパスを入力して下さい")

def photo(cat,x):
	if cat=="1":
	#動物の写真の場合
		pred1=model_bright.predict(x)
		#明暗のモデルを読み込み
		if pred1[0][0] >= pred1[0][1]:
			print("映える")
		else:
			print("映えない")
	elif cat=="2":
	#食べ物の写真の場合
		print("食べ物のカテゴリを選んでください(数字を入力)")
		print("カテゴリを選ばない場合, 精度が著しく低下する可能性があります")
		print("1:一品料理")
		print("2:多品目料理")
		print("3:甘味")
		print("4:その他")
		if cat2=="1":
			pred1=[[0,1,0]]
		elif cat2=="2":
			pred1=[[0,0,1]]
		elif cat2=="3":
			pred1[[1,0,0]]
		else:
		#その他or入力がない場合
			pred1=food_cat.predict(x)
			#食べ物分類用のモデルで強制分類(精度低)
		if np.argmax(pred1)==0:
		#甘味(スイーツ)の写真の場合
			pred2=model_sweets.predict(x)
			#スイーツ映え判別モデルを読み込み
			if pred2[0][0]>=pred2[0][1]:
				print("映える")
			else:
				print("映えない")
		elif np.argmax(pred1)==1:
		#一品料理の写真の場合
			pred2=model_one_food.predict(x)
			#一品料理映えのモデルを読み込み
			if pred2[0][0]>=pred2[0][1]:
				print("映える")
			else:
				print("映えない")
		elif np.argmax(pred1)==2:
		#多品目料理の写真の場合
			pred2=model_course_foods.predict(x)
			#多品目料理映えのモデルを読み込み
			if pred2[0][0]>=pred2[0][1]:
				print("映える")
			else:
				print("映えない")
	elif cat=="3":
	#景色の写真の場合
		pred1=model_bright.predict(x)
		#明暗のモデルを読み込み
		pred2=model_colorful.predict(x)
		#カラフルのモデルを読み込み
		if pred1[0][0] >= pred1[0][1] and pred2[0][0] >= pred2[0][1]:
			print("映える")
		else:
			pred1=model_night.predict(x)
			#夜景のモデルを読み込み
			if pred1[0][1]>=pred1[0][0]:
				print("映える")
			else:
				print("映えない")
	elif cat=="4":
	#建物の画像の場合
		pred1=model_building.predict(x)
		#建物のモデルを読み込み
		if pred1[0][0] >= pred1[0][1]:
			print("映える")
		else:
			print("映えない")
	else:
	#その他or入力がない場合
		cat=np.argmax(model_categories.predict(x))+1
		#カテゴリ分類のモデルを読み込み, 強制的に分類(精度低)
		print("a")
		photo(cat,x)

#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
#予測

print("写真かイラストかを数字で入力してください.")
print("カテゴリを選ばない場合, 精度が著しく低下する可能性があります")
print("1:写真")
print("2:イラスト")
cat=input()
if cat==1:
	preds=[[1,0]]
elif cat==2:
	preds=[[0,1]]
else:
	preds=model_illust.predict(x)


if preds[0][0] > preds[0][1]:
#イラストの場合
	pred1=model_bright.predict(x)
	pred2=model_colorful.predict(x)
	if pred1[0][0] >= pred1[0][1] and pred2[0][0] >= pred2[0][1]:
		print("映える")
	else:
		print("映えない")
else:
#写真の場合
	print("写真のカテゴリを選んでください(数字を入力)")
	print("カテゴリを選ばない場合, 精度が著しく低下する可能性があります")
	print("1:動物")
	print("2:食べ物")
	print("3:景色")
	print("4:建物")
	print("5:その他")
	cat=input()
	photo(cat,x)