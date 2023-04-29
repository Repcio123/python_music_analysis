import math
import random as rn
import numpy as np
import pandas as pd

class DataProcessing:
    @staticmethod
    def shuffle(x):
        for i in range(len(x) - 1, 0, -1):
            j = rn.randint(0, i - 1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def normalization(x, names):
        for column in names:
            data = x.loc[:, column]
            max1 = max(data)
            min1 = min(data)
            for row in range(0, len(x), 1):
                xprim = (x.at[row, column] - min1) / (max1 - min1)
                x.at[row, column] = xprim

    @staticmethod
    def getRating(x):
        lis = [0] * len(x)
        i = 0
        for index, row in x.iterrows():
            lis[i] +=  math.sqrt(row['Views']) + 2 * row['Likes'] / (row['Views']+1) + (row['Comments'] * 0.5) / (row['Views'] +1)
            i+=1
        return lis

    @staticmethod
    def split(x, k):  # k = 0.7, 70% do treningowego
        splitPoint = int(len(x) * k)
        return x.iloc[0:splitPoint], x.iloc[splitPoint:]

    @staticmethod
    def getDistances(x, newObj, columnNames, function, power):
        lis = [0] * len(x)
        i = 0
        for index, row in x.iterrows():
            for column in columnNames:
                lis[i] +=(function(row[column], newObj[column], power))
            i+=1
        return lis

    @staticmethod
    def manhattan(first, second, n):
        return abs(first - second)
                
    @staticmethod
    def euclides(first, second, n):
        return math.pow(math.pow(first, n) - math.pow(second, n), 1/n)

    @staticmethod
    def sort(x, lis):
        x['distance'] = lis
        return x.sort_values(by=['distance'])


    @staticmethod
    def isBanger(x, bar):
        lis = [0]*len(x)
        i = 0
        for index, row in x.iterrows():
            if row['Rating'] >= bar:
                lis[i] = 1
            i+=1
        return lis
            


    @staticmethod
    def NaiveBayes(x, sample, classCol, columnNames):
        classes = x[classCol].unique().tolist()
        
        res = {}
        for var in classes:
            res[var] = []
            for cl in columnNames:
                values = x.loc[x[classCol] == var, cl]
                mean = values.mean()
                sigm2 = values.std()**2
                if sigm2 == 0:
                    print(sigm2)
                    sigm2=0.00000001
                
                res[var].append(DataProcessing.gauss(sample[cl], mean, sigm2))
            res[var] = 1 / len(classes) * np.prod(res[var])
        return max(res, key=res.get)

    @staticmethod
    def bayes(x, sample, classCol, colNames):
        classes = x[classCol].unique().tolist()
        res = {}
        lis = []
        for i in classes:
            lis.append(x[x[classCol]==i])

        for var in lis:
            res[str(var[classCol][0])] = []
            for cl in colNames:
                values = x[cl]
                mean = values.mean()
                sigm2 = values.std()**2
                if sigm2 == 0:
                    sigm2=0.00000001
                
                res[str(var[classCol][0])].append(DataProcessing.gauss(sample[cl], mean, sigm2))
            res[str(var[classCol][0])] = 1 / len(classes) * np.prod(res[str(var[classCol][0])])
        return max(res, key=res.get)
               



    @staticmethod
    def gauss(x, mu, sigm2):
        return (1 / sigm2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigm2)**2)

    @staticmethod
    def useBayes(tS, vS):
        counter = 0
        for i in range(0, len(vS)):
            print(DataProcessing.NaiveBayes(tS, vS.iloc[i], 'variety'), vS.iloc[i]['variety'])
            if DataProcessing.NaiveBayes(tS, vS.iloc[i], 'variety') == vS.iloc[i]['variety']:
                counter += 1
        return counter / len(vS)
