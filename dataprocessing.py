import math
import random as rn
import numpy as np
class DataProcessing:
    @staticmethod
    def shuffle(x):
        for i in range(len(x) - 1, 0, -1):
            j = rn.randint(0, i - 1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def normalization(x):
        values = x.select_dtypes(exclude="object")
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x.loc[:, column]
            max1 = max(data)
            min1 = min(data)
            for row in range(0, len(x), 1):
                xprim = (x.at[row, column] - min1) / (max1 - min1)
                x.at[row, column] = xprim

    @staticmethod
    def split(x, k):  # k = 0.7, 70% do treningowego
        splitPoint = int(len(x) * k)
        return x.iloc[0:splitPoint], x.iloc[splitPoint:]

    @staticmethod
    def getDistances(x, newObj):
        propertiesRange = len(x.columns.values) - 2
        lis = [0] * len(x)
        for row in range(0, len(x), 1):
            for i in range(0, propertiesRange):
                lis[row] += abs(newObj.iat[i] - x.iat[row, i])
                lis[row] += math.pow(newObj.iat[i] - x.iat[row, i], 2)
            lis[row] = math.sqrt(lis[row])
        return lis

    @staticmethod
    def sort(x, lis):
        x['distance'] = lis
        return x.sort_values(by=['distance'])



    @staticmethod
    def NaiveBayes(x, sample, classCol):
        classes = x[classCol].unique().tolist()
        collen = len(x.columns) - 1

        res = {}
        for var in classes:
            res[var] = []
            for cl in x.columns.tolist()[:collen]:
                values = x.loc[x[classCol] == var, cl]
                mean = values.mean()
                sigm2 = values.std()**2
                res[var].append(DataProcessing.gauss(sample[cl], mean, sigm2))
            res[var] = 1 / len(classes) * np.prod(res[var])
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
