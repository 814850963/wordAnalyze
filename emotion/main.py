import numpy as np
import re
import jieba  # 结巴分词
from gensim.models import KeyedVectors
from os import path
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.font_manager as fm
from collections import Counter
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional #时间循环神经网络
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
text_lists = []
sum = pos = neg = 0
model = Sequential()
cn_model = KeyedVectors.load_word2vec_format('/Users/yinhongtao/Desktop/analyze/emotion/sgns.zhihu.bigram', binary=False)
text_result=[]
bg=np.array(Image.open("/Users/yinhongtao/Desktop/analyze/emotion/img/background.jpg"))
class ma:
    def train(self):
        embedding_dim = cn_model['山东大学'].shape[0]
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://blog.csdn.net/qq_40549291/article/details/85274581
        pos_txts = os.listdir('/Users/yinhongtao/Desktop/analyze/emotion/pos')
        neg_txts = os.listdir('/Users/yinhongtao/Desktop/analyze/emotion/neg')
        print('样本总共: ' + str(len(pos_txts) + len(neg_txts)))
        train_texts_orig = []
        train_texts_orig2 = []
        for i in range(len(pos_txts)):
            with open('/Users/yinhongtao/Desktop/analyze/emotion/pos/' + pos_txts[i], 'r', errors='ignore', encoding='GB18030') as f:
                text = f.read().strip()
                train_texts_orig.append(text)
        f.close()
        for i in range(len(neg_txts)):
            with open('/Users/yinhongtao/Desktop/analyze/emotion/neg/' + neg_txts[i], 'r', errors='ignore', encoding='GB18030') as f:
                text = f.read().strip()
                train_texts_orig.append(text)
        f.close()
        train_tokens = []
        for text in train_texts_orig:
            text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
            cut = jieba.cut(text)
            cut_list = [i for i in cut]
            for i, word in enumerate(cut_list):
                try:
                    cut_list[i] = cn_model.vocab[word].index
                except KeyError:
                    cut_list[i] = 0
            train_tokens.append(cut_list)
        num_tokens = [len(tokens) for tokens in train_tokens]
        num_tokens = np.array(num_tokens)
        global max_tokens
        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        max_tokens = int(max_tokens)
        def reverse_tokens(tokens):
            text = ''
            for i in tokens:
                if i != 0:
                    text = text + cn_model.index2word[i]
                else:
                    text = text + ' '
            return text
        reverse = reverse_tokens(train_tokens[0])
        num_words = 50000
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for i in range(num_words):
            embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
        embedding_matrix = embedding_matrix.astype('float32')
        train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                                  padding='pre', truncating='pre')
        train_pad[train_pad >= num_words] = 0
        train_target = np.concatenate((np.ones(2000), np.zeros(2000)))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.1)
        model.add(Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_tokens,
                            trainable=False))
        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        path_checkpoint = '/Users/yinhongtao/Desktop/analyze/emotion/weigths.h5'
        checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                     verbose=1, save_weights_only=True,
                                     save_best_only=True)
        print("Complate")
        try:
            model.load_weights(path_checkpoint)
        except Exception as e:
            earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.1, min_lr=1e-5, patience=0,
                                             verbose=1)
            callbacks = [
                earlystopping,
                checkpoint,
                lr_reduction
            ]

            model.fit(X_train, y_train,
                      validation_split=0.1,
                      epochs=20,
                      batch_size=128,
                      callbacks=callbacks)

            result = model.evaluate(X_test, y_test)
    def predict_sentiment(text):
        print(text)
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0

        print(max_tokens)
        tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
        result = model.predict(x=tokens_pad)
        coef = result[0][0]
        if coef >= 0.5:
            print('是一例正面评价', 'output=%.2f' % coef)
            return '是一例正面评价'

        else:
            print('是一例负面评价', 'output=%.2f' % coef)
            return '是一例负面评价'
    def predict_sentimentp(text):
        mywordList = []
        t = open('/Users/yinhongtao/Desktop/analyze/emotion/codingWord', "r", encoding='utf-8')
        with t:
            data = t.read()
            text_lists = data.split('\n')
        t.close()
        str = ''
        for t in text_lists:
            str += t
        cut = jieba.cut(str)
        listStr = '/'.join(cut)
        f_stop = open('/Users/yinhongtao/Desktop/analyze/emotion/stopword/stopword', encoding="utf8")
        try:
            f_stop_text = f_stop.read()
        finally:
            f_stop.close()
        f_stop_seg_list = f_stop_text.split("\n")
        for myword in listStr.split('/'):
            if not (myword.split()) in f_stop_seg_list and len(myword.strip()) > 1:
                mywordList.append(myword)
        text1 =  ' '.join(mywordList)
        c = Counter()
        for x in mywordList:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
        for (k, v) in c.most_common(20):
            print("%s:%d" % (k, v))
        return c.most_common(20)
    def open(f):
        global sum, pos, neg
        sum = pos = neg = 0
        if f:
            f = open('/Users/yinhongtao/Desktop/analyze/emotion/test/'+f, "r", encoding='utf-8')
            with f:
                data = f.read()
                text_lists = data.split('\n')
            f.close()
            def predict_sentiment(text):
                global sum, pos, neg
                print(text)
                text = re.sub("[^\u4E00-\u9FA5]", "", text)
                cut = jieba.cut(text)
                cut_list = [i for i in cut]
                for i, word in enumerate(cut_list):
                    try:
                        cut_list[i] = cn_model.vocab[word].index
                    except KeyError:
                        cut_list[i] = 0
                tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
                result = model.predict(x=tokens_pad)
                coef = result[0][0]
                sum += coef
                if coef >= 0.5:
                    pos += 1
                else:
                    neg += 1

            for text in text_lists:
                predict_sentiment(text)
            return {'pos':str(pos),'neg':str(neg),'ave':str(sum / (pos + neg) * 100)}

