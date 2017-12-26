# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:01:54 2017

@author: yang
"""

import numpy as np


'''
数据集
'''

critics={
        'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,'Just My Luck': 3.0, 
                      'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
        'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 
                         'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5}, 
        'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                             'Superman Returns': 3.5, 'The Night Listener': 4.0},
        'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,'The Night Listener': 4.5, 
                         'Superman Returns': 4.0, 'You, Me and Dupree': 2.5},
        'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 
                         'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 2.0}, 
        'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,'The Night Listener': 3.0, 
                          'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
        'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}
        }


'''
欧几里得距离度量
在prefs字典中，度量person1和person2之间的距离，从而得到相似度
'''
def sim_distance(prefs, person1, person2):
    sim = {}
    sumOfSqrt = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            sim[item] = 1
            sumOfSqrt += np.power(prefs[person1][item] - prefs[person2][item], 2)
    if len(sim) == 0:
        return 0
    else:
        return 1.0/(1.0 + np.sqrt(sumOfSqrt))


'''
皮卡尔逊度量方法
'''
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    n = len(si)
    if n==0:
        return 1
    sum1 = np.sum([prefs[p1][it] for it in si])
    sum2 = np.sum([prefs[p2][it] for it in si])
    
    sum1Sq = np.sum([np.power(prefs[p1][it], 2) for it in si])
    sum2Sq = np.sum([np.power(prefs[p2][it], 2) for it in si])
    pSum = np.sum([prefs[p1][it] * prefs[p2][it] for it in si])
    
    num = pSum - (sum1*sum2/n)
    den = np.sqrt((sum1Sq - np.power(sum1,2)/n)*(sum2Sq-np.power(sum2,2)/n))
    if den == 0:
        return 0
    r= num/den
    return r


'''
从反映偏好的字典中返回最为匹配者
'''
def topMatches(prefs, person, n=5, similar = sim_pearson):
    scores = [(similar(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


'''
利用所有他人评价值加权平均，为某人提供建议
'''
def getRecommendations(prefs, person, similar = sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        if other == person:
            continue
        sim = similar(prefs, person, other)
        
        if sim <= 0:
            continue
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item]==0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                simSums.setdefault(item, 0)
                simSums[item] += sim
        rankings=[(total/simSums[item],item) for item,total in totals.items()]
        rankings.sort()
        rankings.reverse()
        return rankings


'''
为实现商品推荐，将字典中的人与商品进行交换
'''
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result


'''
构造物品比较数据集
'''
def calculateSimilarItems(prefs, n=10):
    result = {}
    itemPrefs  = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        c += 1
        if c%100==0:
            print('%d / %d' % (c, len(itemPrefs)))
        scores = topMatches(itemPrefs, item, n, similar = sim_distance)
        result[item] = scores
    return result
            

'''
为某人推荐商品
'''
def getRecommendedItems(prefs, itemMatch, user):
    userTatings = prefs[user]
    scores = {}
    totalSim = {}
    
    for (item, rating) in userTatings.items():
        for (similar, item2) in itemMatch[item]:
            if item2 in userTatings:
                continue
            scores.setdefault(item2, 0)
            scores[item2] += similar * rating
            
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similar
    rankings = [(score/totalSim[item], item) for item,score in scores.items()]
    rankings.sort()
    rankings.reverse()
    return rankings


'''
加载电影数据
'''
def loadMovieLens():
    movies = {}
    for line in open(r'F:\Python\CI\2_recommendation\ml-100k/u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title
        
    prefs = {}
    for line in open(r'F:\Python\CI\2_recommendation\ml-100k/u.item/u.data'):
        (user, movieid, rating, ts)=line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs
        







if __name__ == '__main__':
    dis = sim_distance(critics, 'Lisa Rose', 'Gene Seymour')
    print(dis)
    
    dis = sim_pearson(critics, 'Lisa Rose', 'Gene Seymour')
    print(dis)

    score = topMatches(critics, 'Toby', n=3)
    print(score)

    rank = getRecommendations(critics, 'Toby')
    print(rank)

    movie = transformPrefs(critics)
    score = topMatches(movie, 'Superman Returns')
    print(score)

    print('构造物品比较数据集')
    item = calculateSimilarItems(critics)
    print(item)

    print('为某人推荐')
    something = getRecommendedItems(critics, item, 'Toby')
    print(something)

#    data = loadMovieLens()


