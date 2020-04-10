# coding=utf-8


import pandas as pd
import fool
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------------------------------
# 加载停用词词典
stopwords = {}
with open(r'data/stopword.txt', 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords[word.strip()] = 0
# -----------------------------------------------------


# 定义类
class CLF_MODEL:
    # 类目标：该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中

    # 初始化模块
    def __init__(self):
        self.model = ""  # 成员变量，用于存储模型
        self.vectorizer = ""  # 成员变量，用于存储tfidf统计值

    # 训练模块
    def train(self):
        # 函数目标：读取训练数据，训练意图分类模型，并将训练好的分类模型赋值给成员变量self.model
        # input：无
        # output：无

        # 从excel文件读取训练样本
        d_train = pd.read_excel("data/data_train.xlsx")
        # 对训练数据进行预处理
        d_train.sentence_train = d_train.sentence_train.apply(self.fun_clean)
        print("训练样本 = %d" % len(d_train))

        """
        TODO：利用sklearn中的函数进行训练，将句子转化为特征features
        """
        self.vectorizer=TfidfVectorizer(analyzer="word",token_pattern=r"(?u)\b\w+\b")

        self.model=LogisticRegression(C=10)
        features= self.vectorizer.fit_transform(d_train.sentence_train)

        self.model.fit(features, d_train.label)

    # 预测模块（使用模型预测）
    def predict_model(self, sentence):
        # 函数目标：使用意图分类模型预测意图
        #  input：sentence（用户输入）
        # output：clf_result（意图类别），score（意图分数）

        # --------------
        # 对样本中没有的特殊情况做特别判断
        if sentence in ["好的", "需要", "是的", "要的", "好", "要", "是"]:
            return 1, 0.8
        # --------------

        """
        TODO：利用已训练好的意图分类模型进行意图识别
        """
        featurs=self.vectorizer.transform([sentence])
        pre_data = self.model.predict_proba(featurs).tolist()[0]
        clf_result = pre_data.index(max(pre_data))
        score = max(pre_data)

        return clf_result, score

    # 预测模块（使用规则）
    def predict_rule(self, sentence):
        # 函数目标：如果模型训练出现异常，可以使用规则进行预测，同时也可以让学员融合"模型"及"规则"的预测方式
        # input：sentence（用户输入）
        # output：clf_result（意图类别），score（意图分数）

        sentence = sentence.replace(' ', '')
        if re.findall(r'不需要|不要|停止|终止|退出|不买|不定|不订', sentence):
            return 2, 0.8
        elif re.findall(r'订|定|预定|买|购', sentence) or sentence in ["好的","需要","是的","要的","好","要","是"]:
            return 1, 0.8
        else:
            return 0, 0.8

    # 预处理函数
    def fun_clean(self, sentence):
        # 函数目标：预处理函数，将必要的实体转换成统一符号（利于分类准确），去除停用词等
        # input：sentence（用户输入语句）
        # output：sentence（预处理结果）

        """
        预处理函数，将必要的实体转换成统一符号（利于分类准确），去除停用词等
        """
        words ,ners= fool.analysis(sentence)
        ners =ners[0].sort(key=lambda x :len(x[-1]),reverse=True)

        if ners:
            for ner in ners:
                sentence.replace(ner[-1],' '+ner [2] +' ')


        wordslist=fool.cut(sentence)[0]

        wordslist=[word for word in wordslist if word not in stopwords]

        sentence = ' '.join(wordslist)
        sentence= re.sub(r'\s+', ' ', sentence).strip()

        return sentence

    # 分类主函数
    def fun_clf(self, sentence):
        # 函数目标：意图识别主函数
        # input：sentence（ 用户输入语句）
        # output：clf_result（意图类别），score（意图分数）

        # 对用户输入进行预处理
        sentence = self.fun_clean(sentence)
        # 得到意图分类结果（0为“查询”类别，1为“订票”类别，2为“终止服务”类别）
        clf_result, score = self.predict_model(sentence)  # 使用训练的模型进行意图预测
        # clf_result, score = self.predict_rule(sentence)  # 使用规则进行意图预测（可与用模型进行意图识别的方法二选一）
        return clf_result, score


def fun_replace_num(sentence):
    # 函数目标：替换时间中的数字（目的是便于实体识别包fool对实体的识别）
    # input：sentence
    # output：sentence

    # 定义要替换的数字
    time_num = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10","十一":"11","十二":"12"}
    for k, v in time_num.items():
        sentence = sentence.replace(k, v)
    return sentence


def slot_fill(sentence, key=None):
    # 函数目标：填槽函数（该函数从sentence中寻找需要的内容，完成填槽工作）
    # input：sentence（用户输入）, key（指定槽位，只对该句话提取指定槽位的信息）
    # output：slot（返回填槽的结果，以json格式返回，key为槽位名，value为值）

    slot = {}
    # 进行实体识别
    words, ners = fool.analysis(sentence)
    to_city_flag = 0
    for ner in ners[0]:
        if ner[2] == 'time':
            date_content = re.findall(
                r'后天|明天|今天|大后天|周末|周一|周二|周三|周四|周五|周六|周日|本周一|本周二|本周三|本周四|本周五|本周六|本周日|下周一|下周二|下周三|下周四|下周五|下周六|下周日|这周一|这周二|这周三|这周四|这周五|这周六|这周日|\d{,2}月\d{,2}号|\d{,2}月\d{,2}日',
                ner[-1])
            slot["date"] = date_content[0] if date_content else ""

            # 寻找具体时间的关键词
            time_content = re.findall(r'\d{,2}点\d{,2}分|\d{,2}点钟|\d{,2}点', ner[-1])

            pmam_content = re.findall(r'上午|下午|早上|晚上|中午|早晨', ner[-1])

            slot["time"] = pmam_content[0] if pmam_content else "" + time_content[0] if time_content else ""

        if ner[2] == 'location':
            if key is None:
                if re.findall(r'(到|去|回|回去)%s' % (ner[-1]), sentence):
                    to_city_flag = 1
                    slot["to_city"] = ner[-1]
                    continue
                if re.findall(r'从%s|%s出发' % (ner[-1], ner[-1]), sentence):
                    slot["from_city"] = ner[-1]
                elif to_city_flag == 1:
                    slot["from_city"] = ner[-1]
            # 如果指定了槽位
            elif key in ["from_city", "to_city"]:
                slot[key] = ner[-1]
            # 完成出发城市、到达城市的填槽工作

            

    """
    TODO：从sentence中寻找需要的内容，完成填槽工作
    """

    
    return slot


def fun_wait(clf_obj):
    # 函数目标：等待，获取用户输入问句
    # input：CLF_MODEL类实例化对象
    # output：clf_result（用户输入意图类别）, score（意图识别分数）, sentence（用户输入）

    # 等待用户输入
    print("\n\n\n")
    print("-------------------------------------------------------------")
    print("----*------*-----*-----*----*-----*-----*-----*-----*------")
    print("Starting ...")
    sentence = input("客服：请问需要什么服务？(时间请用12小时制表示）\n")
    # 对用户输入进行意图识别
    clf_result, score = clf_obj.fun_clf(sentence)
    return clf_result, score, sentence


def fun_search(clf_result, sentence):
    # 函数目标：为用户查询余票
    # input：clf_result（意图分类结果）, sentence（用户输入问句）
    # output：是否有票

    # 定义槽存储空间
    name = {"time":"出发时间", "date":"出发日期", "from_city":"出发城市", "to_city":"到达城市"}
    slot = {"time":"", "date":"", "from_city":"", "to_city":""}
    # 使用用户第一句话进行填槽
    sentence = fun_replace_num(sentence)
    slot_init = slot_fill(sentence)
    for key in slot_init.keys():
        slot[key] = slot_init[key]
    # 对未填充对槽位，向用户提问，进行针对性填槽
    while "" in slot.values():
        for key in slot.keys():
            if slot[key]=="":
                sentence = input("客服：请问%s是？\n"%(name[key]))
                sentence = fun_replace_num(sentence)
                slot_cur = slot_fill(sentence, key)
                for key in slot_cur.keys():
                    if slot[key]=="":
                        slot[key] = slot_cur[key]

    # 查询是否有票，并答复用户（本次查询是否有票使用随机数完成，实际情况可查询数据库返回）
    if random.random()>0.5:
        print("客服：%s%s从%s到%s的票充足"%(slot["date"], slot["time"], slot["from_city"], slot["to_city"]))
        # 返回1表示有票
        return 1
    else:
        print("客服：%s%s从%s到%s无票" % (slot["date"], slot["time"], slot["from_city"], slot["to_city"]))
        print("End !!!")
        print("----*------*-----*-----*----*-----*-----*-----*-----*------")
        print("-------------------------------------------------------------")
        # 返回0表示无票
        return 0


def fun_book():
    # 函数目标：执行下单订票动作
    # input：无
    # output：无

    print("客服：已为您完成订票。\n\n\n")
    print("End !!!")
    print("----*------*-----*-----*----*-----*-----*-----*-----*------")
    print("-------------------------------------------------------------")



if __name__=="__main__":
    # 实例化对象
    clf_obj = CLF_MODEL()
    # 完成意图识别模型的训练
    clf_obj.train()
    # 用户定义阈值（当分类器分类的分数大于阈值才采纳本次意图分类结果，目的是排除分数过低的意图分类结果）
    threshold = 0.55
    # 循环提供服务
    while 1:
        clf_result, score, sentence = fun_wait(clf_obj)
        # -------------------------------------------------------------------------------
        # 状态转移条件（等待-->等待）：用户输入未达到“查询”、“订票”类别的阈值 OR 意图被分类为“终止服务”
        # -------------------------------------------------------------------------------
        if score<threshold or clf_result==2:
            continue

        # -------------------------------------------------------------------------------
        # 状态转移条件（等待-->查询）：用户输入分类为“查询” OR “订票”
        # -------------------------------------------------------------------------------
        else:
            # 收集订票细节信息
            search_result = fun_search(clf_result, sentence)
            # 查询无票
            # -------------------------------------------------------------------------------
            # 状态转移条件（查询-->等待）：FUN_SEARCH执行完后用户输入意图为“终止服务” OR FUN_SEARCH返回无票
            # -------------------------------------------------------------------------------


            if search_result== 0:
               continue
            else:
                # 等待用户输入
                sentence = input("客服：需要为您订票吗？\n")
                # 对用户输入进行意图识别
                clf_result, score = clf_obj.fun_clf(sentence)
                # -------------------------------------------------------------------------------
                # 状态转移条件（查询-->订票）：FUN_SEARCH返回有票 AND 用户输入意图为“订票”
                # -------------------------------------------------------------------------------
                if clf_result == 1:
                    fun_book()
                    continue

