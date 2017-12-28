# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 19:15:10 2017

@author: yang
"""

import numpy as np
from PIL import Image, ImageDraw
import random

def readfile(filename):
    lines = [line for line in open(filename)]
    #第一行是列标题
    colnames = lines[0].strip().split('\t')
    rownames = []
    data = []
    
    for line in lines[1:]:
        p = line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data
    

'''
皮尔逊距离度量
'''    
def pearson(v1, v2):
    sum1 = np.sum(v1)
    sum2 = np.sum(v2)
    sum1Sq = np.sum([np.power(v,2) for v in v1])
    sum2Sq = np.sum([np.power(v,2) for v in v2])
    
    pSum = np.sum([v1[i]*v2[i] for i in range(len(v1))])
    
    num = pSum - (sum1*sum2/len(v1))
    den = np.sqrt((sum1Sq - np.power(sum1, 2)/len(v1)) * (sum2Sq - np.power(sum2, 2)/len(v1)))
    if den == 0:
        return 0
    return 1.0-num/den
    
    
'''
类 
'''
class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
        
        
'''
分级聚类 
连续不断地将最相似的群组两两合并
'''
def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    while len(clust)>1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)
        
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
                    
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclustid)   
        currentclustid -= 1
        del clust[lowestpair[1]]      
        del clust[lowestpair[0]]
        clust.append(newcluster)   
    return clust[0]        
    
    
'''
用缩进建立层级分布 
'''
def printclust(clust, labels=None, n=0):
    for i in range(n):
        print(' ',end='')
    if clust.id<0:
        print('-',end='')
    else:
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)
    
        
'''
绘制树状图    
'''
def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance


def drawdendrogram(clust, labels, jpeg='clusters.jpg'):
    h = getheight(clust)*20
    w = 1200
    depth=getdepth(clust)

    scaling=float(w-150)/depth

    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    draw.line((0,h/2,10,h/2),fill=(255,0,0))
    
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')


def drawnode(draw,clust,x,y,scaling,labels):
  if clust.id<0:
    h1=getheight(clust.left)*20
    h2=getheight(clust.right)*20
    top=y-(h1+h2)/2
    bottom=y+(h1+h2)/2
   
    ll=clust.distance*scaling
    
    draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))        
    draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))    
    draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))        
   
    drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
    drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
  else:   
    draw.text((x+5,y-7),labels[clust.id],(0,0,0))        
    
    
'''
列聚类    
即将原数据集行列翻转
'''
def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata
    
    
'''
K均值聚类    
'''
def kcluster(rows, distance=pearson, k=4):
    #确定每个维度的最大值和最小值
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
    clusters = [[random.random() * (ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))]
                for j in range(k)]
    
    lastmatches=None
    for t in range(100):
        print('Iteration %d' % t)
        bestmatches = [[] for i in range(k)]
        
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)
        
        if bestmatches==lastmatches:
            break
        lastmatches = bestmatches
        
        for i in range(k):
            avgs = [0.0]*len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    return bestmatches


'''
Tanimoto系数
代表交集与并集的比率
'''
def tanimoto(v1, v2):
    c1, c2, shr = 0, 0, 0
    for i in range(len(v1)):
        if v1[i] != 0:
            c1 += 1
        if v2[i] != 0:
            c2 += 1
        if v1[i] != 0 and v2[i] != 0:
            shr += 1
    return 1.0-(float(shr)/(c1+c2-shr))


'''
多维缩放绘制二维图
para:
    数据集的二维向量
return:
    每个点在图上的坐标
'''
def scaledown(data, distance=pearson, rate=0.01):
    n = len(data)
    realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(n)]
    outersum = 0.0
    loc = [[random.random(), random.random()] for i in range(n)]
    fakedist = [[0.0 for j in range(n)] for i in range(n)]
    
    lasterror = None
    
    #计算任意两点间的当前距离
    for m in range(1000):
        for j in range(n):
            for i in range(n):
                fakedist[i][j] = np.sqrt(np.sum([np.power(loc[i][x] - loc[j][x], 2) for x in range(len(loc[i]))]))
    
        #用于存放每个点的移动量           
        grad = [[0.0, 0.0] for i in range(n)]
        totalerror = 0
    
        #计算目标距离与实际距离的误差
        for k in range(n):
            for j in range(n):
                if j==k:
                    continue
                errorterm = (fakedist[j][k] - realdist[j][k])/realdist[j][k]
            
                grad[k][0] += ((loc[k][0] - loc[j][0])/fakedist[j][k]) * errorterm
                grad[k][1] += ((loc[k][1] - loc[j][1])/fakedist[j][k]) * errorterm
            
                #记录总的误差值
                totalerror += abs(errorterm)
        print(totalerror)
    
        #如果移动之后的效果没有移动之前的误差小，则保存移动之前的误差
        if lasterror and lasterror<totalerror:
            break
        lasterror = totalerror
    
        for k in range(n):
            loc[k][0] -= rate * grad[k][0]
            loc[k][1] -= rate * grad[k][1]
    return loc
    

'''
画出图像
'''
def draw2d(data, labels, jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG') 







    
if __name__ == '__main__':
    print('-------分级聚类------')
    blognames, words, data = readfile(r'F:\Python\CI\3_discovering_group\blogdata.txt')
# =============================================================================
#     clust = hcluster(data)
#     printclust(clust, labels=blognames)
#     
#     drawdendrogram(clust, blognames)
#     
#     print('-------列聚类-------')
#     rdata = rotatematrix(data)
#     clustword = hcluster(rdata)
#     drawdendrogram(clust, words, 'words.jpg')
#     
#     print('-------k均值聚类-------')
#     kclust = kcluster(data, k=10)
#     print(blognames[r] for r in kclust[0])
#     
#     
#     print('-------基于Tanimoto系数的分级聚类-------')
#     wants, people, data1 = readfile(r'F:\Python\CI\3_discovering_group\zebo.txt')
#     clust = hcluster(data1, distance=tanimoto)
#     printclust(clust, labels=wants)
# =============================================================================
    
    print('-------数据点的二维显示-------')
    coords = scaledown(data)
    draw2d(coords, blognames)
    
    
    
    