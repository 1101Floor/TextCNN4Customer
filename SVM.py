# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:56:18 2023

@author: HP
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn import svm
import os
import time
 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

orig_data = pd.read_csv("G:/Customer_TextCNN/data/new_seqdata_cus.csv"
                        , header=0, index_col=0)
orig_data.rename(columns={'class': 'Class'}, inplace=True)

# 分词
sentences_ = []
labels = []
MAX_length = []
for i in np.unique(orig_data.sale_id):
    
    seq = orig_data[orig_data['sale_id'] == i].sentences
    label = orig_data[orig_data['sale_id'] == i].Class
    seq_list = []
    str_num = int(len(seq.values[0]) / 4)
    for j in range(str_num):
        seq_list.append(seq.values[0][j*4:(j+1)*4])
    labels.append(label.values[0])
    MAX_length.append(len(seq_list))
    sentences_.append(seq_list)


#Tokenizer 的示例
tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
tokenizer.fit_on_texts(sentences_)
tensorr = tokenizer.texts_to_sequences(sentences_)
# print(tensorr)

sente_maxlen = 300
key_index = tf.keras.preprocessing.sequence.pad_sequences(tensorr, padding='post',
                                                      maxlen = sente_maxlen, value=0)

train_x, test_x = key_index[:int(6439*0.8),:], key_index[int(6439*0.8):,:]
train_y, test_y = np.array(labels)[:int(6439*0.8)].reshape(-1,1), np.array(labels)[int(6439*0.8):].reshape(-1,1)
train_y, test_y = np.array(train_y), np.array(test_y)
train_y_ = label_binarize(train_y.astype('int64')
                        , classes=[0, 1, 2]).reshape(-1, 3)
test_y_ = label_binarize(test_y.astype('int64')
                        , classes=[0, 1, 2]).reshape(-1, 3)


# SVM
t0 = time.time()
model = svm.SVC(kernel='rbf', probability=True
            , gamma=0.3 ,C=1
            )
y_score = model.fit(train_x, train_y).decision_function(test_x)
# ROC_svm_score = model.predict_proba(test_x)
t1 = time.time()
training_time = t1 - t0
print(training_time)
print("=======================================================")
print("SVM model train accuracy:", model.score(train_x, train_y))
print("SVM model test accuracy:", model.score(test_x, test_y))


def ROC(test_y, y_score):
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if len(test_y.shape) == 3:
        test_y = test_y.reshape(-1, 3)
    else:
        test_y = label_binarize(test_y , classes=[0, 1, 2])        
    for i in range(n_classes): # 遍历三个类别
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc

ROC_svm_score = model.predict_proba(test_x)
svm_fpr, svm_tpr, svm_roc_auc = ROC(test_y_, ROC_svm_score)

target_names = ['NV', 'LV', 'HV']
classification_report_svm = classification_report(list(np.argmax(test_y_, axis=1))
                                                  , list(np.argmax(ROC_svm_score, axis=1))
                                                  , target_names=target_names, digits=4)


alloutdata = np.c_[ROC_svm_score, test_y_]
rocdata = pd.DataFrame(alloutdata)

# rocdata.to_csv('G:/Customer_TextCNN/data/alloutdata_svm.csv')

lw=2
plt.figure(figsize=(5,4),dpi=100)
plt.plot(svm_fpr["micro"],svm_tpr["micro"],
         label='SVM micro-average ROC curve (AUC={0:0.4f})'
               ''.format(svm_roc_auc["micro"]),
         color='cornflowerblue', linestyle=':', linewidth=4)

font1 = {
    # 'family' : 'Times New Roman',
# 'weight' : 'normal',
'size'   : 7,}
font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 10,}
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontdict=font2)
plt.ylabel('True Positive Rate', fontdict=font2)
plt.legend(loc="lower right", prop=font1)
plt.show()

ROC_textcnn_score = model.predict_proba(train_x)

textcnn_fpr, textcnn_tpr, textcnn_roc_auc = ROC(train_y_, ROC_textcnn_score)

target_names = ['NV', 'LV', 'HV']
classification_report_svm_ = classification_report(list(np.argmax(train_y_, axis=1))
                                                  , list(np.argmax(ROC_textcnn_score, axis=1))
                                                  , target_names=target_names, digits=4)