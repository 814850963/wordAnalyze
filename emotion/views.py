import sys
import os
from filecmp import cmp
from django.views.decorators.csrf import csrf_protect
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append('/Users/yinhongtao/Desktop/analyze/venv/lib/python3.7/site-packages/')
import json
from django.http import HttpResponse
from django.shortcuts import render
from .main import *
from .test3beta2 import *
i=0
content = []
def index(request):
    return render(request, 'index.html')
def single(request):
    global i
    print(i)
    if request.method == "GET":
        print(12312321)
        data = (request.GET)
        print(data)
        if(i==0):
            ma.train(0)
            i+=1
        result = ma.predict_sentiment(str(data['word']))
        print(result)
        return HttpResponse(json.dumps({
            "status":1,
            "result":result
        }),content_type='application/json')
    else:
        return HttpResponse(json.dumps({
            "status":0,
            "result":"请求方式错误"
        }),content_type='application/json')
# @csrf_protect
def file(request):
    global i
    if (i == 0):
        ma.train(0)
        i += 1
    if request.method == "POST":
        f = request.FILES.get('file',None)
        destination = open(os.path.join("/Users/yinhongtao/Desktop/analyze/emotion/test", f.name), 'wb+')
        for chunk in f.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        res = ma.open(f.name)
        # return render(request, 'index.html', {'data': data})
        if float(res['ave'])>80:
            a = '优秀'
        elif 60<float(res['ave'])<=80:
            a = '良好'
        elif 50<=float(res['ave'])<=60:
            a = '合格'
        else :
            a = '不合格'
        return HttpResponse(json.dumps({'pos': res['pos'], 'neg': res['neg'], 'ave':a}),content_type='application/json')
def pachong(request):
    global content
    data = request.GET
    city = str(data['city'])
    index = str(data['index'])
    hotelId = str(data['host'])
    content = run(city,index,hotelId)
    return HttpResponse(json.dumps({'arr':content}),content_type='application/json')
def analyze(request):
    global content
    global i
    if (i == 0):
        ma.train(0)
        i += 1
    content1 = copy.deepcopy(content)
    content2 = []
    for t in content1:
        t = re.sub("[^\u4E00-\u9FA5]", "", t)
        content2.append(t)
    f = open('/Users/yinhongtao/Desktop/analyze/emotion/codingWord', 'w')
    f.seek(0)
    f.truncate()
    for i in content2:
        new_context = i + '\n'
        f.write(new_context)
    f.close()
    text1 = ma.predict_sentimentp(content)
    dict = {}
    for (k, v) in text1:  # 输出词频最高的前两个词
        dict[k] = v
    dict = json.dumps(dict)
    return HttpResponse(dict,content_type='application/json')






