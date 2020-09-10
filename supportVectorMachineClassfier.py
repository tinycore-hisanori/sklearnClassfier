from sklearn import tree
import os, glob
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

import pickle

# データが保存されているディレクトリパスの指定(トレーニング用)
train_dir = "./train"

# データが保存されているディレクトリパスの指定(精度確認用)
test_dir = "./test"


X = []
Y = []

#トレーニング用フォルダからバイナリデータを読み込む
image_dir = train_dir
files = glob.glob(image_dir + "/*.bin")
for f in files:
    with open(f, mode='r+b') as fp:
        data = fp.read()
        values = [float(i) for i in data]
        if(len(values) == 256):
            #標準化(0-1の範囲の値にする）
            X1 = [i/255.0 for i in values]
            X.append(X1)
            Y.append(int(os.path.basename(f)[:2]))


trModel = tree.DecisionTreeClassifier()
trModel.fit(X, Y)


#線形分類モデルのサポートベクターマシンによるクラス分類
svm = SVC(kernel='linear', C=1.0, random_state=149)
#学習
svm.fit(X, Y)


#確率的勾配法でのクラス分類
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X, Y)


rfModel = RandomForestClassifier(n_jobs=2, random_state=0)
rfModel.fit(X, Y)

X_TEST = []
LBL = []

image_dir = test_dir
files = glob.glob(image_dir + "/*.bin")
for f in files:
    with open(f, mode='r+b') as fp:
        data = fp.read()
        values = [float(i) for i in data]
        if(len(values) == 256):
            X1 = [i/255.0 for i in values]
            X_TEST.append(X1)
            LBL.append(int(os.path.basename(f)[:2]))


aa = svm.predict(X_TEST)
bb = svm.score(X_TEST, LBL)
print(aa)
print("SVM正答率：" + str(bb))


aa = sgd_clf.predict(X_TEST)
bb = sgd_clf.score(X_TEST, LBL)
print(aa)
print("SGD正答率：" + str(bb))


aa = rfModel.predict(X_TEST)
bb = rfModel.score(X_TEST, LBL)
print(aa)
print("RF正答率：" + str(bb))



aa = trModel.predict(X_TEST)
bb = trModel.score(X_TEST, LBL)
print(aa)
print("TR正答率：" + str(bb))

#学習したモデルを保存する（iOSとかで使える）
filename = 'finalized_model.sav'
pickle.dump(sgd_clf, open(filename, 'wb'))

