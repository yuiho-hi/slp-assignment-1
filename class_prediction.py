import os.path
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



dirname = os.getcwd() #現在のファイルがあるフォルダのパスを取得

#-------------------------------------------------------
#train用プログラム
#-------------------------------------------------------

def logistic(train_file, test_file): #ファイル読み込み用関数
    #引数: (train_file: train指定ファイル)
    with open(train_file, mode='rt') as train_f:
        text_after_train = []
        train_label = []
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        for line in train_f:
            label, text = line.split('\t')
            text_after_train.append(text.replace('\n',''))#preprocessing(text).replace('\n',''))
            train_label.append(label)
        X = vectorizer.fit_transform(text_after_train)
    
    with open(test_file, mode='rt') as test_f:
        text_after_test = []
        test_label = []
        for line in test_f:
            label, text = line.split('\t')
            text_after_test.append(text.replace('\n',''))#preprocessing(text).replace('\n',''))
            test_label.append(label)
        Y = vectorizer.transform(text_after_test)

    lr = LogisticRegression(max_iter=10000).fit(X, np.array(train_label))
    train_label_pred = lr.predict(X)
    test_label_pred = lr.predict(Y)

    print(f'精度 (train): {accuracy_score(train_label, train_label_pred)}')
    print(f'精度 (test): {accuracy_score(test_label, test_label_pred)}')

    print(f'適合率 micro (test): {precision_score(test_label, test_label_pred, average="micro")}')
    print(f'適合率 macro (test): {precision_score(test_label, test_label_pred, average="macro")}')

    print(f'再現率 micro (test): {recall_score(test_label, test_label_pred, average="micro")}')
    print(f'再現率 macro (test):　{recall_score(test_label, test_label_pred, average="macro")}')

    print(f'F1スコア micro (test): {f1_score(test_label, test_label_pred, average="micro")}')
    print(f'F1スコア macro (test): {f1_score(test_label, test_label_pred, average="macro")}')



logistic(os.path.join(dirname, "titles-en-train.labeled"), os.path.join(dirname, "titles-en-test.labeled")) #学習　trainデータとn-gram数を渡す


