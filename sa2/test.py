import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib #jbolib模块
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
def save_model(filepath):
    res = pd.read_csv('data/data_train.csv')
    res_y= pd.read_csv('data/label_train.csv')
    res["label"]=res_y["label"]
    print(res_y.shape)
    print(res.shape)
    res.dropna(axis=0, how='any', inplace=True)
    res = res.drop(['indexs'], axis=1)
    tag=res['label'].values
    res = res.drop(['label'], axis=1)
    #

    xtrain, xtest, ytrain, ytest = train_test_split(res.values, tag)
    print("开始训练数据")
    model = LogisticRegression(penalty="l1", C=100, solver='liblinear')
    model.fit(xtrain, ytrain)
    predict_value = model.predict(xtest)
    scores = cross_val_score(model, xtrain, ytrain, cv=5)

    print(scores)
    joblib.dump(model, 'data/logistic_clf.pkl')

# 获取名称列表
def busness_name_get(filepath):
    busness_name = {}

    with open(filepath, encoding='utf-8') as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            busness_name[data["business_id"]]=data["name"]
            line = f.readline()
    print("获取名称成功")
    return busness_name

pre_model = joblib.load('data/logistic_clf.pkl')
#预测方法
def predict_prob(vectors):
    features = np.array(vectors, dtype=np.float16)
    proba_value = pre_model.predict_proba(features)
    score = proba_value[:, 1]
    return np.float16(score[0])
_ = glove2word2vec('data/glove.6B.100d.txt', 'data/glove2word2vec.6B.100d.txt')
model = KeyedVectors.load_word2vec_format('data/glove2word2vec.6B.100d.txt')


def docvec_get(sentence):
    # print("获取维度")
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


def pre_data(busness_name):
    i = 1
    with open("data/result.json", "a+", encoding='utf-8') as refile:
        with open("data/business_review_aspect_segment.json", encoding='utf-8') as file:
            line = file.readline()
            while line:
                data = json.loads(line)
                ratingsum = 0
                data_temp = {}
                business_id = data["business_id"]
                aspect_data = data["aspect_data"]
                result_seg={}
                for key in aspect_data.keys():
                    tmp = {}
                    posdata = {}
                    pos = []
                    negdata = {}
                    neg = []
                    for item in aspect_data[key]:
                        content = item["text"]
                        vetor = docvec_get(content)
                        if  not np.isnan(vetor).any():
                            score = predict_prob(vetor)

                            if score > 0.5:
                                posdata[content] = score
                            else:
                                negdata[content] = score
                    sorted_pos = sorted(posdata.items(), key=lambda x: x[1], reverse=False)
                    for _, item in enumerate(sorted_pos):
                        if len(pos) < 5:
                            pos.append(item[0])

                    sorted_neg = sorted(negdata.items(), key=lambda x: x[1], reverse=True)
                    for _, item in enumerate(sorted_neg):
                        if len(neg) < 5:
                            neg.append(item[0])
                    rating = round((float(len(posdata.items())) / (len(posdata.items())+len(negdata.items()))), 1)
                    ratingsum = ratingsum + rating
                    tmp["rating"] = rating
                    tmp["pos"] = pos
                    tmp["neg"] = neg
                    result_seg[key] = tmp
                ratings = round((float(ratingsum) / len(result_seg.items())), 1)
                business_name = busness_name[business_id]
                data_temp["business_id"] = business_id
                data_temp["name"] = business_name
                data_temp["rating"] = ratings
                data_temp["Detailed Rating"] = result_seg
                refile.write(json.dumps(data_temp) + "\n")
                line = file.readline()
                print("处理了%d条数据" % i)
                i = i + 1
            print("保存完成")

def get_segment_detail(busness_id):
    with open("data/result.json",  encoding='utf-8') as refile:
        line =refile.readline()
        while line :
            if busness_id in line :
                return line
            else:
                line=refile.readline()



def main():
    # save_model("")
    # 加载模型

    # print("加载模型成功")
    # busness_name=busness_name_get('data/filter_business.json')
    # pre_data(busness_name)
    detail=get_segment_detail("NyLYY8q1-H3hfsTwuwLPCg")
    print(detail)
if __name__ == "__main__":
    main()