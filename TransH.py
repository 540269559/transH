import os
import codecs
import numpy as np
import copy
import time
import random

entity2id = {}
relation2id = {}
relationTPH = {}
relationHPT = {}
# filePath = "./FB15k/"
filePath = "./WN18/"
# 将进程运行的内容文件删除
def clearProgressText(filePath='./progressMessage/', fileName='trainStatus.txt'):
    file = filePath + fileName
    if os.path.exists(file):
        os.remove(file)

# 将进程运行的内容输入到text中
def writingProgressText(text, filePath='./progressMessage/', fileName='trainStatus.txt'):
    file = filePath + fileName
    with open(file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

# 存储结果
def storeResult(fileName, vectorDict, filePath='./trainResult/'):
    file = filePath + fileName
    with open(file, 'w', encoding='utf-8') as f:
        for key in vectorDict:
            f.write(key + "\t")
            f.write(str(list(vectorDict[key])))
            f.write("\n")

# 删除训练结果的文件
# def clearTrainResult():
#     directory = "./trainResult"
#     # 获取目录下的所有文件和子目录
#     files_in_directory = os.listdir(directory)
#     # 遍历目录下的所有文件和子目录
#     for item in files_in_directory:
#         # 构建文件或子目录的完整路径
#         full_path = os.path.join(directory, item)

#         # 判断是否为文件，如果是则删除
#         if os.path.isfile(full_path):
#             os.remove(full_path)
#     return

# 加载数据
def data_loader(file):
    entityFileName = file + 'entity2id.txt'
    relationFileName = file + 'relation2id.txt'
    with open(entityFileName) as entityF, open(relationFileName) as relationF:
        entityLines = entityF.readlines()
        relationLines = relationF.readlines()
        for line in entityLines:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            entityName = arr[0]
            entityId = arr[1]
            entity2id[entityName] = entityId
        for line in relationLines:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            relationName = arr[0]
            relationId = arr[1]
            relation2id[relationName] = relationId
    # 加载训练数据的文件
    relationHead = {}
    relationTail = {}
    tripleList = []
    trainFileName = file + 'train.txt'
    with open(trainFileName) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            headEntityName = arr[0]
            relationName = arr[1]
            tailEntityName = arr[2]
            
            headEntityId = entity2id[headEntityName]
            tailEntityId = entity2id[tailEntityName]
            relationId = relation2id[relationName]
            tripleList.append([headEntityId, relationId, tailEntityId])

            # 记录当前关系边的头节点的个数
            if relationId in relationHead:
                if headEntityId in relationHead[relationId]:
                    relationHead[relationId][headEntityId] += 1
                else:
                    relationHead[relationId][headEntityId] = 1
            else:
                relationHead[relationId] = {}
                relationHead[relationId][headEntityId] = 1
            # 记录当前关系边的尾节点的个数
            if relationId in relationTail:
                if tailEntityId in relationTail[relationId]:
                    relationTail[relationId][tailEntityId] += 1
                else:
                    relationTail[relationId][tailEntityId] = 1
            else:
                relationTail[relationId] = {}
                relationTail[relationId][tailEntityId] = 1
    # sum1记录记录边的头节点种类数
    # sum2记录边对应节点的个数
    for relation in relationHead:
        sum1, sum2 = 0, 0
        for head in relationHead[relation]:
            sum1 += 1
            sum2 += relationHead[relation][head]
        # 这条边平均每个头节点会出现多少次
        tph = sum2 / sum1
        relationTPH[relation] = tph
    for relation in relationTail:
        sum1, sum2 = 0, 0
        for tail in relationTail[relation]:
            sum1 += 1
            sum2 += relationTail[relation][tail]
        # 这条边平均每个头节点会出现多少次
        hpt = sum2 / sum1
        relationHPT[relation] = hpt
    entityIdSet = set(entity2id.values())
    relationIdSet = set(relation2id.values())
    message = "所有数据加载完毕,实体有%d个,关系有%d个,三元组有%d个" % (len(entityIdSet), len(relationIdSet), len(tripleList))
    print(message)
    writingProgressText(message)
    return entityIdSet, relationIdSet, tripleList

# 正则化
def normalization(vector):
    return vector / np.linalg.norm(vector)
    
# 第一范式  h,r,t属于同一平面
# def norm_l1(h, r, t):
#     np.linalg.norm(h + r - t, ord=1)

# 计算h与t关于在r所处平面的距离
def distance(hVector, rNormVector, rHyperVector, tVector):
    # h垂直平面向量
    normH = np.dot(hVector, rNormVector) * rNormVector
    # h在平面上的向量
    hyperH = hVector - normH
    # 同理t也一样
    normT = np.dot(tVector, rNormVector) * rNormVector
    hyperT = tVector - normT
    # 此时则是要求在超平面的h+r≈t
    resultVector = hyperH + rHyperVector - hyperT
    
    """
    天坑!!!
    np.linalg.norm(resultVector)表示第二范数
    np.sum(np.square(resultVector))表示将向量的每个数平方相加得出结果
    """
    # return np.linalg.norm(resultVector)
    return np.sum(np.square(resultVector))

# 这个可以先不管
# def scale_entity(self, h, t, h_c, t_c):

class TransH:
    def __init__(self, entitySet, relationSet, tripleList, dimension=50, 
                lr=0.01, margin=1.0, norm=1, C=1.0, epsilon = 1e-5):
        
        self.entitySet = entitySet 
        self.relationSet = relationSet 
        self.tripleList = tripleList 
        self.dimension = dimension 
        self.lr = lr 
        self.margin = margin 
        self.norm = norm 
        self.C = C 
        # 防止分母为零的小数，通常取一个很小的正数，如1e-5
        self.epsilon = epsilon 

        # 损失值
        self.loss = 0

        # 实体向量
        # id:Vector
        self.entityVector = {}
        # 关系向量
        self.relationVector = {}

        # transH算法需要用到的两个关系边的向量
        # 1.关于超平面的法向量
        self.relationNormVector = {}
        # 2.关于超平面上的单位向量
        self.relationHyperVector = {}
        

    # 初始化向量
    def dataInitialization(self):
        high = 6.0 / np.sqrt(self.dimension)
        low = -6.0 / np.sqrt(self.dimension)
        for entityId in self.entitySet:
            self.entityVector[entityId] = np.random.uniform(low, high, self.dimension)

        for relation in self.relationSet:
            # 注意这里需要正则化,因为我们需要的是平面上的单位法向量,还有平面上的任意向量
            vector1 = normalization(np.random.uniform(low, high, self.dimension))
            vector2 = normalization(np.random.uniform(low, high, self.dimension))
            self.relationNormVector[relation] = vector1
            self.relationHyperVector[relation] = vector2
        

    # epochs轮数
    # batch批次
    def training_run(self, epochs=10, batch=400):
        # 根据批次获得每次训练样本的总数
        batchSize = int(len(self.tripleList) / batch)
        # 先按轮次
        for epoch in range(epochs):
            # 对实体向量进行正则化
            for entityId in self.entityVector:
                self.entityVector[entityId] = normalization(self.entityVector[entityId])
            self.loss = 0.0
            # 记录每一轮的时间
            startTime = time.time()
            count = 0

            # 洗牌样本列表
            random.shuffle(self.tripleList)

            for i in range(batch):
                tripleSamples = self.tripleList[i * batchSize: (i + 1) * batchSize]
                # tripleSamples = random.sample(self.tripleList, batchSize)
                # 最后用于更新的数据,里面包含负样本
                finalTriples = []
                
                count += 1
                if count % 20 == 0:
                    message = "当前批次为:%d,总共批次为:%d" % (count, batch)
                    writingProgressText(message)
                    print(message)
                for triple in tripleSamples:

                    copyTriple = copy.deepcopy(triple)
                    relationId = copyTriple[1]
                    """
                    这里关于p的说明 
                    tph 表示每一个尾节点对应的平均头节点数 
                    hpt 表示每一个头节点对应的平均尾结点数
                    当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体
                    """
                    num = np.random.random(1)[0]
                    p = relationTPH[relationId] / (relationTPH[relationId] + relationHPT[relationId])
                    # 修改头部
                    if p > num:
                        copyTriple[0] = random.sample(self.entityVector.keys(), 1)[0]
                        while copyTriple[0] == triple[0]:
                            copyTriple[0] = random.sample(self.entityVector.keys(), 1)[0]
                    # 修改尾部
                    else:
                        copyTriple[2] = random.sample(self.entityVector.keys(), 1)[0]
                        while copyTriple[2] == triple[2]:
                            copyTriple[2] = random.sample(self.entityVector.keys(), 1)[0]
                    if (triple, copyTriple) not in finalTriples:
                        finalTriples.append((triple, copyTriple))
                self.update_triple_embedding(finalTriples)
                
            endTime = time.time()
            spendTime = round(endTime - startTime, 2)
            message = "总轮数:%d,当前进行的轮数:%d,本轮花费的时间:%.2f,损失值:%d" % (epochs, epoch + 1, spendTime, self.loss)
            print(message)
            writingProgressText(message)
        # 存储结果
        entityFileName = "entity_" + filePath + "_" + str(self.dimension) + "dim_batch" + str(batch)
        relationNormFileName = "relation_norm_" + filePath + "_" + str(self.dimension) + "dim_batch" + str(batch)
        relationHyperFileName = "relation_hyper_" + filePath + "_" + str(self.dimension) + "dim_batch" + str(batch)
        storeResult(entityFileName, self.entityVector)
        storeResult(relationNormFileName, self.relationNormVector)
        storeResult(relationHyperFileName, self.relationHyperVector)


    # 另一种loss计算方式
    # def orthogonality(self, norm, hyper):
    # 最重要的更新函数
    def update_triple_embedding(self, sample):
        
        copyEntity2Vector = copy.deepcopy(self.entityVector)
        copyRelationNormVectorDict = copy.deepcopy(self.relationNormVector)
        copyRelationHyperVectorDict = copy.deepcopy(self.relationHyperVector)

        for realTriple, virtualTriple in sample:
            # 拿到五个id
            # 两个头两个尾一个关系
            realHeadId = realTriple[0]
            relationId = realTriple[1]
            realTailId = realTriple[2]
            virtualHeadId = virtualTriple[0]
            virtualTailId = virtualTriple[2]

            # 拿到六个复制的对应的向量
            copyRealHeadVector = copyEntity2Vector[realHeadId]
            copyRealTailVector = copyEntity2Vector[realTailId]
            copyVirtualHeadVector = copyEntity2Vector[virtualHeadId]
            copyVirtualTailVector = copyEntity2Vector[virtualTailId]
            copyRelationNormVector = copyRelationNormVectorDict[relationId]
            copyRelationHyperVector = copyRelationHyperVectorDict[relationId]

            # 拿到六个原生的对应的向量
            realHeadVector = self.entityVector[realHeadId]
            realTailVector = self.entityVector[realTailId]
            virtualHeadVector = self.entityVector[virtualHeadId]
            virtualTailVector = self.entityVector[virtualTailId]
            relationNormVector = self.relationNormVector[relationId]
            relationHyperVector = self.relationHyperVector[relationId]

            # 计算向量之间的距离
            # 真实距离
            realDistance = distance(realHeadVector, relationNormVector, relationHyperVector, realTailVector)
            # 负样本距离
            virtualDistance = distance(virtualHeadVector, relationNormVector, relationHyperVector, virtualTailVector)  
            
            # 计算损失值
            loss = self.margin + realDistance - virtualDistance
            if loss > 0:
                self.loss += loss
                # 单位向量
                normVector = np.ones(self.dimension)
                # 距离公式 
                # distance = (((h - norm^T * h * norm )+ hyper - (t - norm^T * t * norm )))^2
                # 超平面上的h+r-t
                realHyperHRT = (
                    (realHeadVector - np.dot(relationNormVector, realHeadVector) * relationNormVector) 
                    + relationHyperVector 
                    - (realTailVector - np.dot(relationNormVector, realTailVector) * relationNormVector)
                )
                virtualHyperHRT = (
                    (virtualHeadVector - np.dot(relationNormVector, virtualHeadVector) * relationNormVector)
                    + relationHyperVector
                    - (virtualTailVector - np.dot(relationNormVector, virtualTailVector) * relationNormVector)
                )
                # 对h求导
                realGradient = 2 * realHyperHRT * (normVector - relationNormVector ** 2)
                # 对h'求导
                virtualGradient = 2 * virtualHyperHRT * (normVector - relationNormVector ** 2)
                # 对平面向量求导
                hyperGradient = 2 * realHyperHRT - 2 * virtualHyperHRT
                # 对平面法向量求导
                normGradient = (
                    2 * realHyperHRT * (realTailVector - realHeadVector) * 2 * relationNormVector
                    - 2 * virtualHyperHRT * (virtualTailVector - virtualHeadVector) * 2 * relationNormVector
                )
                
                # 梯度求完之后就开始更新
                realGradientRate = self.lr * realGradient
                normGradientRate = self.lr * normGradient
                hyperGradientRate = self.lr * hyperGradient
                virtualGradientRate = self.lr * virtualGradient

                # 先更新真实的向量
                copyRealHeadVector -= realGradientRate
                copyRealTailVector += realGradientRate
                copyRelationNormVector -= normGradientRate
                copyRelationHyperVector -= hyperGradientRate

                # 如果替换的是头部
                if realHeadId != virtualHeadId:
                    copyVirtualHeadVector += virtualGradientRate
                    copyRealTailVector -= virtualGradientRate
                elif realTailId != virtualTailId:
                    copyRealHeadVector += virtualGradientRate
                    copyVirtualTailVector -= virtualGradientRate

                copyEntity2Vector[realHeadId] = normalization(copyRealHeadVector)
                copyEntity2Vector[realTailId] = normalization(copyRealTailVector)
                # 更新负样本的节点值
                # 如果替换的是头部,头部的负样本值要更新,反之同理
                if realHeadId != virtualHeadId:
                    copyEntity2Vector[virtualHeadId] = normalization(virtualHeadVector)
                elif realTailId != virtualTailId:
                    copyEntity2Vector[virtualTailId] = normalization(virtualTailVector)

                copyRelationNormVectorDict[relationId] = normalization(copyRelationNormVector)
                copyRelationHyperVectorDict[relationId] = copyRelationHyperVector
        self.entityVector = copyEntity2Vector
        self.relationNormVector = copyRelationNormVectorDict
        self.relationHyperVector = copyRelationHyperVectorDict

                

if __name__ == '__main__':
    # 删除进行是的输出文件
    clearProgressText()
    # 删除结果文件
    # clearTrainResult()
    entityIdSet, relationIdSet, tripleList = data_loader(filePath)

    transH = TransH(entityIdSet, relationIdSet, tripleList, dimension=50, lr=0.01, margin=1.0, norm=1)
    transH.dataInitialization()
    transH.training_run(epochs=100)



