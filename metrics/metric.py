import math
import  numpy as np
from math import log


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return float(error) / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(float(error) / count)

    @staticmethod
    def calDCG_k(dictdata, k):
        DCG = []
        iDCG = []
        nDCG = []
        for key in dictdata.keys():
            listdata = dictdata[key]
            real_value_list = sorted(listdata, key=lambda x: x[0], reverse=True)
            # print(real_value_list)
            idcg = 0.0
            predict_value_list = sorted(listdata, key=lambda x: x[1], reverse=True)
            # print(predict_value_list)
            dcg = 0.0
            if len(listdata) >= k:
                for i in range(k):
                    idcg += (pow(2, real_value_list[i][0]) - 1) / (log(i + 2, 2))
                    dcg += (pow(2, predict_value_list[i][0]) - 1) / (log(i + 2, 2))
                iDCG.append(idcg)
                DCG.append(dcg)
            else:
                continue
                # min_len = len(listdata)
                # for i in range(min_len):
                #     idcg += (pow(2, real_value_list[i][0]) - 1) / (log(i + 2, 2))
                #     dcg += (pow(2, predict_value_list[i][0]) - 1) / (log(i + 2, 2))
                # iDCG.append(idcg)
                # DCG.append(dcg)
        # print(iDCG)
        # print(DCG)
        # for i in iDCG:
        #     if i == 0:
        #         print(i)
        for i in range(len(DCG)):
            nDCG.append(DCG[i] / iDCG[i])
        # ave_dcg = np.sum(DCG) / len(DCG)
        # ave_idcg = np.sum(iDCG) / len(iDCG)
        ave_ndcg = np.sum(nDCG) / len(nDCG)
        # print(nDCG)
        return ave_ndcg

