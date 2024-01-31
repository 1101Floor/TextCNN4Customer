import tensorflow as tf
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import time

orig_data = pd.read_csv("F://Customer_TextCNN//data//new_seqdata_cus1202.csv"
                        , header=0, index_col=0)
orig_data.rename(columns={'class': 'Class', 'customer': 'sale_id'}, inplace=True)
# 分词
sentences_ = []
labels = []
MAX_length = []
for i in np.unique(orig_data.sale_id):
    
    seq = orig_data[orig_data['sale_id'] == i].sentences
    label = orig_data[orig_data['sale_id'] == i].Class
    seq_list = []
    str_num = int(len(seq.values[0]) / 3)
    for j in range(str_num):
        seq_list.append(seq.values[0][j*3:(j+1)*3])
    labels.append(label.values[0])
    MAX_length.append(len(seq_list))
    sentences_.append(seq_list)
        

# 训练Word2Vec模型

word_vector_len = 100
sente_maxlen = 300

t0 = time.time()
model = Word2Vec(sentences_, vector_size=word_vector_len, window=5
                 , sg=0, hs=1
                 , min_count=1, workers=4)
# 获取词向量
word_list = sentences_
sente = []
# vectors = []
for word in word_list:
    # print(word)
    key_index = []
    for j in word:
        # print(j)
        key_index.append(model.wv.key_to_index[j])
    # print(key_index)
    # sente.append(key_index)
    key_index = tf.keras.preprocessing.sequence.pad_sequences([key_index], padding='post',
                                                      maxlen = sente_maxlen, value=99)
    vectors = []
    # print(word)
    for j in key_index[0]:
        if j < len(model.wv.index_to_key):
            vectors.append(model.wv[j])
        else:
            vectors.append([0] * 100)
    # print(np.array(vectors).shape)
    sente.append(vectors)

train_x, test_x = np.array(sente)[:int(6439*0.8),:,:], np.array(sente)[int(6439*0.8):,:,:]
train_y, test_y = np.array(labels)[:int(6439*0.8)].reshape(-1,1), np.array(labels)[int(6439*0.8):].reshape(-1,1)
train_y, test_y = np.array(train_y), np.array(test_y)
train_y = label_binarize(train_y.astype('int64')
                        , classes=[0, 1, 2]).reshape(-1, 3)
test_y = label_binarize(test_y.astype('int64')
                        , classes=[0, 1, 2]).reshape(-1, 3)

class TextCNN(tf.keras.models.Model):
    def __init__(self, drop_rate, cnn_num, label_dim):
        super().__init__()
        self.drop_rate = drop_rate
        self.cnn_num = cnn_num
        self.label_dim = label_dim
        self.logits_conv1d = {}
        self.logits_maxpooling1d = {}
        self.conv1d, self.maxpooling1d = {}, {}
        self.inputlayer = tf.keras.layers.InputLayer()
        self.dropout = tf.keras.layers.Dropout(rate=self.drop_rate)
        
        for i in range(self.cnn_num):
            self.conv1d[i] = tf.keras.layers.Conv1D(filters=64, kernel_size= i+2
                                                    , activity_regularizer=tf.keras.regularizers.l2(0.05) # 输出正则化
                                                    , kernel_regularizer=tf.keras.regularizers.l2(0.05))
            self.maxpooling1d[i] = tf.keras.layers.MaxPooling1D(pool_size=10)
        
        # self.attention = tf.keras.layers.Attention()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=10)
        self.Dense = tf.keras.layers.Dense(units = self.label_dim, activation='softmax')
        self.Dense_1 = tf.keras.layers.Dense(units = 128)
        self.acti = keras.layers.ReLU()
    
    
    def call(self, inputs):
        print(inputs.shape,'inputs')
        for i in range(self.cnn_num):
            self.logits_conv1d[i] = self.conv1d[i](inputs)
            print(self.logits_conv1d[i].shape,'conv1d')
            
            self.logits_maxpooling1d[i] = self.maxpooling1d[i](self.logits_conv1d[i])
            print(self.logits_maxpooling1d[i].shape,'maxpooling1d')
        maxpooling1d_list = [self.logits_maxpooling1d[j] for j in range(self.cnn_num)]
        
        contanct_output = tf.keras.layers.concatenate(maxpooling1d_list, axis=-1)
        flatCnn = keras.layers.Flatten()(contanct_output)
        print(flatCnn.shape, 'contanct')
        desen1 = self.Dense_1(flatCnn)
        print(desen1.shape)
        dropout = self.dropout(desen1)
        densen1Relu = self.acti(dropout)
        output = self.Dense(densen1Relu) 
        print(output.shape, 'output')
        return output

epochs=500
batch_size = 322

encoder_input = tf.keras.layers.Input(shape=([sente_maxlen, word_vector_len]))

output = TextCNN(drop_rate = 0.5
                           , cnn_num = 4
                           , label_dim = 3
                           )(encoder_input)

model_textcnn_att = tf.keras.models.Model([encoder_input], [output])
model_textcnn_att.summary()
model_textcnn_att.compile(loss = keras.losses.CategoricalCrossentropy()
              ,optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
              ,metrics = tf.keras.metrics.CategoricalAccuracy())

history = model_textcnn_att.fit([train_x], train_y, epochs=epochs
          , batch_size = batch_size
          , validation_data=([test_x], test_y))

t1 = time.time()
training_time = t1 - t0
print(training_time)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(framealpha=0.01)
# plt.xlabel('Epochs %s batch_size %s' % (epochs, batch_size), fontsize = 12)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('CBOW+TextCNN Training Loss', fontsize = 12)
plt.show()
plt.plot(history.history['categorical_accuracy'], label='train')
plt.plot(history.history['val_categorical_accuracy'], label='test')
plt.legend(framealpha=0.01)
# plt.xlabel('Epochs %s batch_size %s' % (epochs, batch_size), fontsize = 12)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('CBOW+TextCNN Training Accuracy', fontsize = 12)
plt.show()

Loss_acc = np.c_[history.history['loss'],history.history['val_loss']
                 ,history.history['categorical_accuracy'],history.history['val_categorical_accuracy']]

loss_acc = pd.DataFrame(Loss_acc, columns=['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy'])

# loss_acc.to_csv('G:/Customer_TextCNN/data/cobw_loss_acc.csv')

# 评估指标
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

ROC_cobw_score = model_textcnn_att.predict(test_x)

cobw_fpr, cobw_tpr, cobw_roc_auc = ROC(test_y, ROC_cobw_score)

target_names = ['NV', 'LV', 'HV']
classification_report_cobw = classification_report(list(np.argmax(test_y, axis=1))
                                                  , list(np.argmax(ROC_cobw_score, axis=1))
                                                  , target_names=target_names, digits=4)


alloutdata = np.c_[ROC_cobw_score, test_y]
rocdata = pd.DataFrame(alloutdata)

# rocdata.to_csv('G:/Customer_TextCNN/data/alloutdata_cobw.csv')

lw=2
plt.figure(figsize=(5,4),dpi=100)
plt.plot(cobw_fpr["micro"],cobw_tpr["micro"],
         label='CBOW+TextCNN micro-average ROC curve (AUC={0:0.4f})'
               ''.format(cobw_roc_auc["micro"]),
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

ROC_textcnn_score = model_textcnn_att.predict(train_x)

textcnn_fpr, textcnn_tpr, textcnn_roc_auc = ROC(train_y, ROC_textcnn_score)

target_names = ['NV', 'LV', 'HV']
classification_report_textcnn_ = classification_report(list(np.argmax(train_y, axis=1))
                                                  , list(np.argmax(ROC_textcnn_score, axis=1))
                                                  , target_names=target_names, digits=4)