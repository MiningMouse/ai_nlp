import pandas as pd
import numpy as py
import json
import csv

busness_ids=[]
#首先过滤掉评论少于100的
#将过滤掉的内容存储为filter_business.json
def filter_review_Data(filepath):
    i = 0
    with open("data/filter_business.json", "w+", encoding='utf-8') as csvfile:
        with open(filepath, encoding='utf-8') as f:
            line = f.readline()
            while line:

                data = json.loads(line)
                if data["review_count"] > 100:
                    csvfile.write(line)
                    i = i + 1
                    print("写入%d条数据" % i)
                line = f.readline()

#返过滤后数据的business_id总和
def busness_id_get(filepath):
    busness_ids = []

    with open(filepath, encoding='utf-8') as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            busness_ids.append(data["business_id"])
            line = f.readline()
    return busness_ids

#讲评论数据进行过滤，只留下评论高于100的
#同时内容按照business_id进行整合
#相同business_id的评论内容结合到一个json对象中
#保存为business_review.json
def business_review_get(busness_ids,filepath):
    busness_review = {}
    with open(filepath, encoding='utf-8') as f:
        line = f.readline()
        while len(line) > 2:
            data = json.loads(line)
            if data['business_id'] in busness_ids and data['business_id'] in busness_review.keys():
                temp = {}
                temp["review_id"] = data['review_id']
                temp["stars"] = data['stars']
                temp["text"] = data['text']
                busness_review[data['business_id']].append(temp)
            elif data['business_id'] in busness_ids:
                temp = {}
                temp["review_id"] = data['review_id']
                temp["stars"] = data['stars']
                temp["text"] = data['text']
                busness_review[data['business_id']] = [temp]
            line = f.readline()

    with open("data/business_review.json", "a+", newline='', encoding='utf-8') as csvfile:
        for key in busness_review.keys():

            temp = {"business_id": key, "review_list": busness_review[key]}
            # print(json.dumps(temp))
            csvfile.write(json.dumps(temp) + "\n")
    print("插入成功")


# 获取aspect的list集合保存数据
import nltk
import json

# 获取aspect的list集合保存数据
#数据保存business_review_aspect.json

def get_aspect_busness(filepath):
    with open("data/business_review_aspect.json", "a+", newline='', encoding='utf-8') as csvfile:
        f = open("data/business_review.json")
        # 返回一个文件对象
        line = f.readline()
        i = 1
        # 调用文件的 readline()方法
        while line:
            data = json.loads(line)
            busnessid = data["business_id"]
            aspects_dic = {}
            for item in data["review_list"]:
                sentence = item["text"]
                tagged_words = []
                tokens = nltk.word_tokenize(sentence)
                tag_tuples = nltk.pos_tag(tokens)
                for (word, tag) in tag_tuples:
                    if tag == "NN":
                        if word not in aspects_dic:
                            aspects_dic[word] = [item]
                        else:
                            aspects_dic[word].append(item)

            aspects_sort = sorted(aspects_dic.items(), key=lambda x: len(x[1]), reverse=True)
            aspects_dic = {}

            for _, item in enumerate(aspects_sort):

                if len(aspects_dic.items()) < 5:
                    aspects_dic[item[0]] = item[1]
                else:
                    break

            temp = {}
            temp["business_id"] = data["business_id"]
            temp["aspect_data"] = aspects_dic
            csvfile.write(json.dumps(temp) + "\n")
            line = f.readline()
            i=i+1
            print("处理了%d条" % i)


import json
import nltk


def get_cur_aspect_adj(text):
    tokens = nltk.word_tokenize(text)
    tag_tuples = nltk.pos_tag(tokens)
    for (word, tag) in tag_tuples:
        if tag == "JJ" or tag == "ADJ":
            return word
    return None

# 根据过滤出的top5的aspect进行过滤，只保留aspect top5中数据
#保存为business_review_aspect_segment.json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize
def get_aspect_segments(filepath):
    with open("data/business_review_aspect_segment.json","a+",encoding='utf-8') as file:
        with open("data/business_review_aspect.json",encoding='utf-8') as csvfile:
            i = 1
            line = csvfile.readline()
            while line :
                data = json.loads(line.strip())
                business_id=data["business_id"]
                aspect_data=data["aspect_data"]
                for key in aspect_data.keys():
                    for item in aspect_data[key] :
                        content =item["text"]
                        for s in sent_tokenize(content):
                            if key in s:
                                content=s
                                break
                        item["text"]= content
                file.write(json.dumps(data)+"\n")
                print("处理了%d条" % i)
                line =csvfile.readline()
                i = i+1


from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
import numpy as np

_ = glove2word2vec('data/glove.6B.100d.txt', 'data/glove2word2vec.6B.100d.txt')
model = KeyedVectors.load_word2vec_format('data/glove2word2vec.6B.100d.txt')


def docvec_get(sentence):
    print("获取维度")
    """
    将分词数据转为句向量。
    seg: 分词后的数据

    return: 句向量
    """
    seg = nltk.word_tokenize(sentence)
    vector = np.zeros((1, 100))
    size = len(seg)
    for word in seg:
        try:
            vector += model.wv[word]
        except KeyError:
            size -= 1

    return vector / size


import numpy as np
#准备训练数据 star大于4的为正样本 小于2为负样本
#同时文本数据进行向量化表示 方便训练
def business_review_get(filepath):
    x_train = []
    y_train=[]
    busness_review = {}
    i=1
    with open(filepath, encoding='utf-8') as f:
        line = f.readline()
        print("处理数据")
        while len(line) > 2 and i< 500000:
            data = json.loads(line)

            if data['stars'] > 4:
                tag = 1
                vertor = docvec_get(data['text'])
                y_train.append(tag)
                x_train.append(vertor.flatten())
            elif data['stars'] < 2:
                tag = 0
                vertor = docvec_get(data['text'])
                y_train.append(tag)
                x_train.append(vertor.flatten())
            print("处理了%d条" % i)
            i=i+1
            line = f.readline()
    data_train = pd.DataFrame(x_train)
    datay_train = pd.DataFrame(y_train)
    data_train.to_csv("data/data_train.csv")
    print("保存成功")
    datay_train.to_csv("data/label_train.csv")
    print("label保存成功")
def main():
    # filter_review_Data("data/business.json")
    # busness_ids = busness_id_get("data/filter_business.json")
    # business_review_get(busness_ids,"G:/BaiduNetdiskDownload/yelp_dataset/yelp_dataset/review.json")
    # get_aspect_busness("")
    # get_aspect_segments("")

    business_review_get("G:/BaiduNetdiskDownload/yelp_dataset/yelp_dataset/review.json")


if __name__ == "__main__":


	main()


