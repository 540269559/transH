# -*- coding: utf-8 -*-
import json
import operator
import torch
import time

from TransH import clearProgressText, writingProgressText

entity2id = {}
relation2id = {}

def loadEntityAndRelation(filePath):
    entityFile = filePath + "entity2id.txt"
    relationFile = filePath + "relation2id.txt"
    with open(entityFile) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            entityName = arr[0]
            entityId = arr[1]
            entity2id[entityName] = entityId

    with open(relationFile) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            relationName = arr[0]
            relationId = arr[1]
            relation2id[relationName] = relationId


def load_vector(fileName, filePath='./trainResult/'):
    file = filePath + fileName
    resultVectorDict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            resultVectorDict[arr[0]] = json.loads(arr[1])
    return resultVectorDict

def load_triple(fileName, filePath='./FB15k/'):
    file = filePath + fileName
    resultTripleList = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue
            headName = arr[0]
            relationName = arr[1]
            tailName = arr[2]
            headId = entity2id[headName]
            relationId = relation2id[relationName]
            tailId = entity2id[tailName]
            resultTripleList.append([headId, relationId, tailId])
    return resultTripleList
class testTransH:
    def __init__(self, entityVectorDict, relationNormVectorDict, relationHyperVectorDict, testTripleList, trainTripleList, validTripleList, filter_triple=False ,norm=1):
        self.entityVectorDict = entityVectorDict 
        self.relationNormVectorDict = relationNormVectorDict 
        self.relationHyperVectorDict = relationHyperVectorDict 
        self.testTripleList = testTripleList 
        self.trainTripleList = trainTripleList 
        self.validTripleList = validTripleList 
        self.filter = filter_triple 
        self.norm = norm 

    def test_run(self):
        hits = 0
        rank_sum = 0
        
        count = 0

        # 测试集合
        for triple in self.testTripleList:
            startTime = time.time()
            count += 1
            
            rank_head_dict = {}
            # key:三元组，value：距离distance
            rank_tail_dict = {}

            self.headFilter = []
            self.tailFilter = []

            # 是否过滤
            # 这里的过滤主要是指构造负样本的时候，
            # 如果这个负样本已经在测试集合里面了，那就过滤掉
            if self.filter:
                self.createNegativeSample(testTripleList, triple)
                self.createNegativeSample(trainTripleList, triple)
                self.createNegativeSample(validTripleList, triple)
            
            
            # 所有实体替换头部，构造新的三元组并计算距离存储到rank数组中
            self.computeDistanceAndStoreResult(triple, rank_head_dict)
            # 替换尾部
            self.computeDistanceAndStoreResult(triple, rank_tail_dict, head=False)
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            # 根据距离从小到大排序
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
            # 开始记录
            for i in range(len(rank_head_sorted)):
                rankTriple = rank_head_sorted[i][0]
                if triple[0] == rankTriple[0]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            for i in range(len(rank_tail_sorted)):
                rankTriple = rank_head_sorted[i][0]
                if triple[2] == rankTriple[2]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            endTime = time.time()
            spendTime = round(endTime - startTime, 2)
            if count % 100 == 0:
                print(hits, rank_sum)
                message = "当前正在测试数量:%d,总数量为:%d" % (count, len(self.testTripleList))
                print(message)
                writingProgressText(message, fileName='./testStatus.txt')
                message = ("本次花费时间为:%.2fs,前十排名率为%.2f,总排名平均为%.2f" 
                            % (spendTime, hits / (2 * count), rank_sum / (2 * count))
                          )
                print(message)  
                writingProgressText(message, fileName='./testStatus.txt')
        # 除以二是因为头尾各一次
        reuslt_hit_10 = hits / (2 * len(self.testTripleList))
        reuslt_mean_rank = rank_sum / (2 * len(self.testTripleList))
        return reuslt_hit_10, reuslt_mean_rank
    # 计算正负样本的距离并存储到rank数组中
    def computeDistanceAndStoreResult(self, triple, rankHeadDict, head = True):
        head_embedding = []
        tail_embedding = []
        norm_relation = []
        hyper_relation = []
        temp = []
        # 所有实体替换头部，构造新的三元组
        for index, entityId in enumerate(self.entityVectorDict.keys()):
            if head:
                newTriple = [entityId, triple[1], triple[2]]
            else:
                newTriple = [triple[0], triple[1], entityId]
            # print(newTriple)
            if self.filter and newTriple in self.headFilter:
                continue
            head_embedding.append(self.entityVectorDict[newTriple[0]])
            tail_embedding.append(self.entityVectorDict[newTriple[2]])
            norm_relation.append(self.relationNormVectorDict[newTriple[1]])
            hyper_relation.append(self.relationHyperVectorDict[newTriple[1]])
            temp.append(tuple(newTriple))
        distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)
        for i in range(len(temp)):
            rankHeadDict[temp[i]] = distance[i]

    # 构造负样本
    def createNegativeSample(self, tripleList, targetTriple):
        for triple in tripleList:
            if targetTriple[2] == triple[2] and targetTriple[1] == triple[1] and targetTriple[0] != triple[0]:
                self.headFilter.append(triple)
            if targetTriple[0] == triple[0] and targetTriple[1] == triple[1] and targetTriple[2] != triple[2]:
                self.tailFilter.append(triple)
        return

    # 计算距离，注意，这个时候传入的参数都是list类型
    def distance(self, h, r_norm, r_hyper, t):
        # 使用 torch.from_numpy() 将 NumPy 数组转换为 PyTorch 张量
        hVector = torch.tensor(h)
        rNormVector = torch.tensor(r_norm)
        rHyperVector = torch.tensor(r_hyper)
        tVector = torch.tensor(t)
        
        """
        torch.sum(input, dim=None, keepdim=False, dtype=None)
        input:输入的张量
        dim:沿着哪个维度进行求和，不指定dim，则整个张量进行求和
        dim=0按行相加、dim=1按列相加
        keepdim:是否保持输出张量的维度和输入张量相同

        hVector * rNormVector表示两个张量的对应位置的数字相乘，就是点乘
        torch.sum()让他们按列相加，得到一个数值，然后乘以rNormVector向量，完美！
        """
        hHyperVector = hVector - torch.sum(hVector * rNormVector, dim = 1, keepdim=True) * rNormVector
        tHyperVector = tVector - torch.sum(tVector * rNormVector, dim = 1, keepdim=True) * rNormVector
        
        distanceHT = hHyperVector + rHyperVector - tHyperVector
        score = torch.norm(distanceHT, p=self.norm, dim=1)
        return score.numpy()

if __name__ == "__main__":
    # 初始化entity2id和relation2id
    loadEntityAndRelation("./FB15k/")
    
    # 处理一下文本记录文件
    clearProgressText(fileName='./testStatus.txt')
    entitySet = set()
    relationSet = set()
    # 需要加载的文件:
    # 1.实体向量
    entityVectorDict = load_vector('entity_50dim_batch400')
    # 2.关系法向量
    relationNormVectorDict = load_vector('relation_norm_50dim_batch400')
    # 3.关系平面向量
    relationHyperVectorDict = load_vector('relation_hyper_50dim_batch400')
    # 4.训练文件的三元组
    trainTripleList = load_triple('train.txt')
    # 5.验证文件的三元组
    validTripleList = load_triple('valid.txt')
    # 6.测试文件的三元组
    testTripleList = load_triple('test.txt')
    message = "加载完毕，实体向量个数:%d,关系法向量个数:%d,关系平面向量个数:%d" % (len(entityVectorDict.keys()), len(relationNormVectorDict.keys()), len(relationHyperVectorDict.keys()))
    print(message)
    writingProgressText(message, fileName='./testStatus.txt')
    message = "训练三元组个数:%d,验证三元组个数:%d,测试三元组个数:%d" % (len(trainTripleList), len(validTripleList), len(testTripleList))
    print(message)
    writingProgressText(message, fileName='./testStatus.txt')
    # 把这些三元组全部拿到，用于test_run验证

    test = testTransH(
        entityVectorDict, relationNormVectorDict, 
        relationHyperVectorDict, testTripleList, 
        trainTripleList, validTripleList, 
        filter_triple=False, norm=2)
    test.test_run()
    hit10, mean_rank = test.test_run()
    print("raw entity hits@10: ", hit10)
    print("raw entity meanrank: ",mean_rank)
    result = "hits@10:%.4f, meanrank:%.4f" % (hit10, mean_rank)
    writingProgressText(result, filePath='', fileName='./result.txt')
