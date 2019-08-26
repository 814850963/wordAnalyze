import re
import copy
import json
import requests
import pandas as pd
from xpinyin import Pinyin
author=[]
date = []
content = []
score=[]
global index
index=1
global city
city='1'
global hotelId
hotelId='3452'
global url
def crawlFunction(city,index,hotelId):
    global url
    global content
    if index == 1:
        content = []
        url = "http://touch.qunar.com/api/hotel/hoteldetail/comment?seq=" + city + "_city_" + hotelId
        crawExe(1)
    if index > 1:
        crawlFunction(city,1,hotelId)
        for i in range(2,index):
            url = "http://touch.qunar.com/api/hotel/hoteldetail/comment?seq=" + city + "_city_" + hotelId + "&commentType=0&commentPage="+str(i)
            crawExe(i)
def crawExe(index):
    # for i in range(1, index+1):÷
        try:
            print("正在抓取第" + str(index) + "页")
            print(url)
            html = requests.get(url).text
            print(html)
            html = json.loads(html)
            data = html['data']
            print(data)
            commentData = data['commentData']
            comments = commentData['comments']
            allTotal = commentData['allTotal']
            goodTotal = commentData['goodTotal']
            mediumTotal = commentData['mediumTotal']
            badTotal = commentData['badTotal']
            avgscore = commentData['score']
            hotelName = commentData['hotelName']
            for each in comments:
                # print(each)
                author1=each['author']
                date1=each['date']
                content1 = each['content']
                score1=each['score']
                author.append(author1)
                date.append(date1)
                score.append(score1)
                content.append(content1)
        except:
            pass

def run(city,index,hotelId):
    p = Pinyin()
    city=p.get_pinyin(city, '')
    crawlFunction(city,int(index),hotelId)

    return content


