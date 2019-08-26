# 首先加载必用的库
from emotion.mainWindow import *

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QThread,pyqtSignal,QFile
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import chardet
import jieba  # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
# 使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib


matplotlib.use("Qt5Agg")  # 声明使用QT5
neg = pos = sum = 0
max_tokens = 0
text_lists = []
model = Sequential()
cn_model = KeyedVectors.load_word2vec_format("./sgns.zhihu.bigram", binary=False)
text_result = []

class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # Form
        self.F = MyFigure(width=3,height=2,dpi=100)

        self.TrainButton.clicked.connect(self.train)  # 训练按钮
        self.reset.clicked.connect(self.input.clear)  # 重置
        self.analyse.clicked.connect(self.Analyse)  # 分析

        # RadioButtonGroup
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.addButton(self.radio_openfile)
        self.buttonGroup.addButton(self.radio_input)
        self.buttonGroup.addButton(self.radio_net)

        self.radio_openfile.toggled.connect(self.opTest)
        self.radio_input.toggled.connect(self.inTest)
        self.radio_net.toggled.connect(self.netTest)
        self.frame_openfile.setVisible(False)
        self.frame_opResult.setVisible(False)
        

        # 读取文件
        self.open_txt.clicked.connect(self.open)
        self.open_csv.clicked.connect(self.open)
        self.open_json.clicked.connect(self.open)
        self.open_sql.clicked.connect(self.open)

        self.tabWidget.setCurrentIndex(0)


    def inTest(self):
        self.frame_input.setVisible(True)
        self.frame_inputResult.setVisible(True)
        self.frame_openfile.setVisible(False)
        self.frame_opResult.setVisible(False)
    def opTest(self):
        self.frame_input.setVisible(False)
        self.frame_inputResult.setVisible(False)
        self.frame_openfile.setVisible(True)
        self.frame_opResult.setVisible(True)
    def netTest(self):
        self.frame_inputResult.setVisible(False)
        self.frame_input.setVisible(False)
        self.frame_openfile.setVisible(False)
    def open(self):
        global sum,pos,neg
        sum = pos = neg = 0
        if self.sender() == self.open_txt:
            file = QFileDialog.getOpenFileName(
                self, "打开文件", "./", "txt文件(*.txt)")

            if file[0]:
                f = QFile(file[0])
                f = open(file[0],"r",encoding='utf-8')
                with f:
                    data = f.read()
                    text_lists=data.split('\n')

                f.close()
            self.OutputBox.append("---------txt文件内容----------")
            def predict_sentiment(text):
                global sum,pos,neg
                print(text)
                # 除去除中文外所有字符
                text = re.sub("[^\u4E00-\u9FA5]", "", text)
                self.OutputBox.append('数据清洗后：'+text)
                # 分词
                cut = jieba.cut(text)
                cut_list = [i for i in cut]
                # tokenize
                for i, word in enumerate(cut_list):
                    try:
                        cut_list[i] = cn_model.vocab[word].index
                    except KeyError:
                        cut_list[i] = 0
                # padding
                tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
                # 预测
                result = model.predict(x=tokens_pad)
                coef = result[0][0]
                sum+=coef
                if coef >= 0.5:
                    pos+=1
                else:
                    neg+=1

            for text in text_lists:
                self.OutputBox.append('-'+text)
                predict_sentiment(text)
            self.sum_show.setText(str(len(text_lists)))
            self.pos_show.setText(str(pos))
            self.neg_show.setText(str(neg))
            # self.aver_show.setText(str("%.1f"%(sum/(pos+neg)))) 
            self.aver_show.setText('%d'%(sum/(pos+neg)*100)+'/100') 
            
        elif self.sender() == self.open_csv:
            file = QFileDialog.getOpenFileName(
                self, "打开文件", "./", "Csv files(*.csv)")

            if file[0]:
                f = QFile(file[0])
                f = open(file[0],"r",encoding='utf-8')
                with f:
                    data = f.read()
                    text_lists=data.split('\n')

                f.close()
            self.OutputBox.append("---------CSV文件内容----------")
            def predict_sentiment(text):
                global sum,pos,neg
                print(text)
                # 除去除中文外所有字符
                text = re.sub("[^\u4E00-\u9FA5]", "", text)
                self.OutputBox.append('数据清洗后：'+text)
                # 分词
                cut = jieba.cut(text)
                cut_list = [i for i in cut]
                # tokenize
                for i, word in enumerate(cut_list):
                    try:
                        cut_list[i] = cn_model.vocab[word].index
                    except KeyError:
                        cut_list[i] = 0
                # padding
                tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
                # 预测
                result = model.predict(x=tokens_pad)
                coef = result[0][0]
                sum+=coef
                if coef >= 0.5:
                    pos+=1
                else:
                    neg+=1
    
            for text in text_lists:
                self.OutputBox.append(text)
                predict_sentiment(text)
                
            self.sum_show.setText(str(len(text_lists)))
            self.pos_show.setText(str(pos))
            self.neg_show.setText(str(neg))
            self.aver_show.setText('%d'%(sum/(pos+neg)*100)+'/100')
        elif self.sender() == self.open_sql:
            file = QFileDialog.getOpenFileName(
                self, "打开文件", "./", "SQL files(*.sql)")
        else:
            file = QFileDialog.getOpenFileName(
                self, "打开文件", "./", "JSON files(*.json)")

            if file[0]:
                f = QFile(file[0])
                f = open(file[0],"r",encoding='utf-8')
                with f:
                    data = json.loads(f.read())
                    text_lists=np.array(data['comment'])

                f.close()
            self.OutputBox.append("---------JSON文件内容----------")
            
            def predict_sentiment(text):
                global sum,pos,neg
                
                print(text)
             
                # 除去除中文外所有字符
                text = re.sub("[^\u4E00-\u9FA5]", "", text)
                self.OutputBox.append('数据清洗后：'+text)
                # 分词
                cut = jieba.cut(text)
                cut_list = [i for i in cut]
                # tokenize
                for i, word in enumerate(cut_list):
                    try:
                        cut_list[i] = cn_model.vocab[word].index
                    except KeyError:
                        cut_list[i] = 0
                # padding
                tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
                # 预测
                result = model.predict(x=tokens_pad)
                coef = result[0][0]
                sum+=coef
                if coef >= 0.5:
                    pos+=1
                else:
                    neg+=1
             
            for index in range(len(text_lists)):
                self.OutputBox.append(str(text_lists[index]))
                predict_sentiment(str(text_lists[index]))
          
            self.sum_show.setText(str(len(text_lists)))
            self.pos_show.setText(str(pos))
            self.neg_show.setText(str(neg))
            self.aver_show.setText('%d'%(sum/(pos+neg)*100)+'/100')

    def train(self):
        self.progressBar.setValue(10)
        warnings.filterwarnings("ignore")

        # 由此可见每一个词都对应一个长度为300的向量
        embedding_dim = cn_model['山东大学'].shape[0]
        import os

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        pos_txts = os.listdir('./pos')
        neg_txts = os.listdir('./neg')
        # print('样本总共: ' + str(len(pos_txts) + len(neg_txts)))
        self.OutputBox.append('样本总共: ' + str(len(pos_txts) + len(neg_txts)))
    
        train_texts_orig = []  # 存储所有评价，每例评价为一条string
        train_texts_orig2 = []  # 存储所有评价，每例评价为一条string

        for i in range(len(pos_txts)):
            with open('pos/' + pos_txts[i], 'r', errors='ignore', encoding='GB18030') as f:
                text = f.read().strip()
                # print(text)
                train_texts_orig.append(text)
        f.close()
        self.progressBar.setValue(20)
        # print(train_texts_orig)
        for i in range(len(neg_txts)):
            with open('neg/' + neg_txts[i], 'r', errors='ignore', encoding='GB18030') as f:
                text = f.read().strip()
                # print(text)
                train_texts_orig.append(text)
        f.close()
        self.progressBar.setValue(30)
        train_tokens = []
        for text in train_texts_orig:
            # 去掉标点
            text = re.sub(
                "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
            # 结巴分词
            cut = jieba.cut(text)
            # 结巴分词的输出结果为一个生成器
            # 把生成器转换为list
            cut_list = [i for i in cut]

            for i, word in enumerate(cut_list):
                try:
                    # 将词转换为索引index
                    cut_list[i] = cn_model.vocab[word].index
                except KeyError:
                    # 如果词不在字典中，则输出0
                    cut_list[i] = 0
            train_tokens.append(cut_list)

        self.progressBar.setValue(34)
        # 获得所有tokens的长度
        num_tokens = [len(tokens) for tokens in train_tokens]
        num_tokens = np.array(num_tokens)
        global max_tokens
        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        max_tokens = int(max_tokens)

    
        self.F.axes.hist(np.log(num_tokens), bins=100)
        self.F.fig.suptitle("Distribution of tokens length")
        self.progressBar.setValue(40)
        # 用来将tokens转换为文本
        def reverse_tokens(tokens):
            text = ''
            for i in tokens:
                if i != 0:
                    text = text + cn_model.index2word[i]
                else:
                    text = text + ' '
            return text
        self.progressBar.setValue(48)
        reverse = reverse_tokens(train_tokens[0])

   
        num_words = 50000
        # 初始化embedding_matrix，之后在keras上进行应用
        embedding_matrix = np.zeros((num_words, embedding_dim))
        self.progressBar.setValue(55)
        for i in range(num_words):
            embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
        embedding_matrix = embedding_matrix.astype('float32')

        train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                                  padding='pre', truncating='pre')

        # 超出五万个词向量的词用0代替
        train_pad[train_pad >= num_words] = 0

        self.progressBar.setValue(60)

        train_target = np.concatenate((np.ones(2000), np.zeros(2000)))

        from sklearn.model_selection import train_test_split

        
        
        # 90%的样本用来训练，剩余10%用来测试
        X_train, X_test, y_train, y_test = train_test_split(
            train_pad, train_target, test_size=0.1)
        self.progressBar.setValue(69)
        model.add(Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_tokens,
                            trainable=False))
        self.progressBar.setValue(77)

        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))

        model.add(Dense(1, activation='sigmoid'))
        # 我们使用adam以0.001的learning rate进行优化
        optimizer = Adam(lr=1e-3)
        self.progressBar.setValue(80)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        self.progressBar.setValue(88)
        model.summary()
        path_checkpoint = './weigths.h5'
        checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                     verbose=1, save_weights_only=True,
                                     save_best_only=True)

    
        self.OutputBox.append("Complate")
        self.progressBar.setValue(100)
        self.analyse.setEnabled(True)
        self.TrainButton.setText("训练完成")
        self.TrainButton.setEnabled(False)
        self.progressBar.setEnabled(False)
        # 尝试加载已训练模型
        
        model.load_weights(path_checkpoint)
        self.testlayout.addWidget(self.F,0,1)
        

    def Analyse(self):
        self.OutputBox.append('输入：'+self.input.text())
        warnings.filterwarnings("ignore")
        text = self.input.text()
        print(text)
        # 去除除中文外所有字符
        text = re.sub("[^\u4E00-\u9FA5]", "", text)
        # 分词
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0
        # padding
        tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
        # 预测
        result = model.predict(x=tokens_pad)
        coef = result[0][0]

        if coef >= 0.75:
            print('Positive', 'rate=%.2f' % coef)
            self.OutputBox.append('Positive'+' rate=%.2f' % coef)
            self.judge.setText("该评价是一例正面评价")
            self.rate.setText("客户满意度约为："+'%d'%(coef*100)+'/100')
            self.graphicsView.setStyleSheet("background:url(:/img/Excellent.png) no-repeat; border:n")
        elif coef >= 0.55:
            print('Positive', 'rate=%.2f' % coef)
            self.OutputBox.append('Positive'+' rate=%.2f' % coef)
            self.judge.setText("该评价是一例正面评价")
            self.rate.setText("客户满意度约为："+'%d'%(coef*100)+'/100')
            self.graphicsView.setStyleSheet("background:url(:/img/Good.png) no-repeat; border:n")
        elif coef >= 0.45:
            print('Objective', 'rate=%.2f' % coef)
            self.OutputBox.append('Objective'+' rate=%.2f' % coef)
            self.judge.setText("该评价是一例折中的评价")
            self.rate.setText("客户满意度约为："+'%d'%(coef*100)+'/100')
            self.graphicsView.setStyleSheet("background:url(:/img/Average.png) no-repeat; border:n")
        elif coef >= 0.3:
            print('Positive', 'rate=%.2f' % coef)
            self.OutputBox.append('Positive'+' rate=%.2f' % coef)
            self.judge.setText("该评价是一例负面评价")
            self.rate.setText("客户满意度约为："+'%d'%(coef*100)+'/100')
            self.graphicsView.setStyleSheet("background:url(:/img/Fair.png) no-repeat; border:n")
        else:
            print('Negative', 'rate=%.2f' % coef)
            self.OutputBox.append('Negative'+' rate=%.2f' % coef)
            self.judge.setText("该评价是一例负面评价")
            self.rate.setText("客户满意度约为："+'%d'%(coef*100)+'/100')
            self.graphicsView.setStyleSheet("background:url(:/img/Poor.png) no-repeat; border:n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
