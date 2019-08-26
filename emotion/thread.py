from os import path

from pandas.tests.extension.numpy_.test_numpy_nested import np
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.font_manager as fm
bg=np.array(Image.open("/Users/yinhongtao/Desktop/analyze/emotion/img/background.jpg"))
def runmain(self):
    t = open('/Users/yinhongtao/Desktop/analyze/emotion/Word', "r", encoding='utf-8')
    with t:
        data = t.read()
        text_lists = data.split('\n')
    t.close()
    # 分词
    str = []
    for t in text_lists:
        str.append(t)
    text1 = ' '.join(str)
    wc = WordCloud(
            background_color="white",
            max_words=20,
            mask=bg,  # 设置图片的背景
            max_font_size=60,
            random_state=42,
            font_path='/Users/yinhongtao/Desktop/analyze/emotion/youyuan.ttf'  # 中文处理，用系统自带的字体
        ).generate(text1)
    # 为图片设置字体
    my_font = fm.FontProperties(fname='/Users/yinhongtao/Desktop/analyze/emotion/youyuan.ttf')
    # 产生背景图片，基于彩色图像的颜色生成器
    image_colors = ImageColorGenerator(bg)
    # 开始画图
    plt.imshow(wc.recolor(color_func=image_colors))
    # 为云图去掉坐标轴
    plt.axis("off")
    # 画云图，显示
    # plt.figure()
    plt.show()
    # 为背景图去掉坐标轴
    plt.axis("off")
    plt.imshow(bg, cmap=plt.cm.gray)
    # 保存云图
    wc.to_file("/Users/yinhongtao/Desktop/analyze/emotion/ciyun_02.png")
