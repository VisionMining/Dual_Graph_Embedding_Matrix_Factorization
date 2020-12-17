import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict
from metrics.metric import Metric
from utility.tools import denormalize
from reader.rating import RatingGetter
from configx.configx import ConfigX
import pickle
from utility.tools import sigmoid
import math
from metrics.measure import Measure
from utility.tools import normalize
from sklearn.metrics import roc_auc_score
import scores
class MF(object):
    """
    docstring for MF
    the base class for matrix factorization based model-parent class

    """

    def __init__(self):
        super(MF, self).__init__()
        self.config = ConfigX()
        self.rg = RatingGetter()  # loading raing data
        # self.init_model()
        self.iter_rmse = []
        self.iter_mae = []
        self.iter_ndcg = []
        self.test = np.zeros(self.rg.max_u*self.rg.max_i)
        self.valid = np.zeros(self.rg.max_u*self.rg.max_i)
        self.predict_test = np.zeros(self.rg.max_u*self.rg.max_i)
        self.predict_valid = np.zeros(self.rg.max_u*self.rg.max_i)
        self.test_data = np.zeros((self.rg.max_u, self.rg.max_i))
        self.valid_data = np.zeros((self.rg.max_u, self.rg.max_i))
        pass

    def init_model(self):
        self.P = np.random.rand(self.rg.max_u, self.config.factor) / (
        self.config.factor ** 0.5)  # latent user matrix
        self.Q = np.random.rand(self.rg.max_i, self.config.factor) / (
        self.config.factor ** 0.5)  # latent item matrix
        # self.Bu = np.random.rand(self.rg.get_train_size()[0])  # bias value of user
        # self.Bi = np.random.rand(self.rg.get_train_size()[1])  # bais value of item
        print(self.rg.max_u)
        print(self.rg.max_i)
        print(self.rg.get_train_size())
        # print(self.rg.id2user[4243])
        print(len(self.rg.id2item))
        self.loss, self.lastLoss = 0.0, 0.0
        self.lastRmse, self.lastMae, self.lastNdcg = 10.0, 10.0, 0.0
        pass

    def train_model(self):
        pass

#################################################################################################
    # # 将list转为dict，只保存了list数据中的user_id 和 rating数据
    # def list2dic(self, listData):
    #     dic = defaultdict(list)
    #     for i in listData:
    #         dic[i[0]].append(i[2])
    #     return dict(dic)
    #
    # # 将实际的rating和预测的rating合并成一个,ratingData 和 predict_ratingdata都是列表
    # def merge_r_pr(self, ratingData, predict_ratingdata):
    #     dic = defaultdict(list)
    #     dic_ratingdata = self.list2dic(ratingData)
    #     dic_predictdata = self.list2dic(predict_ratingdata)
    #     key_list = list(dic_ratingdata.keys())
    #     for i in range(len(key_list)):
    #         for length in range(len(dic_ratingdata[key_list[i]])):
    #             dic[key_list[i]].append([dic_ratingdata[key_list[i]][length], dic_predictdata[key_list[i]][length]])
    #     return dic
    #
    # def valid_model(self):
    #     res = []
    #     list_pre = []
    #     list_true = []
    #     u_r_pr = defaultdict(list)
    #     for ind, entry in enumerate(self.rg.validSet()):
    #         user, item, rating = entry
    #         # predict
    #         pred = self.predict(user, item)
    #         # denormalize
    #         # prediction = denormalize(prediction, self.config.min_val, self.config.max_val)
    #
    #         # pred = self.checkRatingBoundary(prediction)
    #         # add prediction in order to measure
    #         # self.dao.testData[ind].append(pred)
    #         res.append([user, item, rating, pred])
    #         list_pre.append([user, item, pred])
    #         list_true.append([user, item, rating])
    #     rmse = Metric.RMSE(res)
    #     mae = Metric.MAE(res)
    #     sort_predata = sorted(list_pre, key=lambda x: (x[0], x[1]))
    #     sort_truedata = sorted(list_true, key=lambda x: (x[0], x[1]))
    #     u_r_pr = self.merge_r_pr(sort_truedata, sort_predata)
    #     ndcg = Metric.calDCG_k(u_r_pr, self.config.k)
    #
    #     self.iter_rmse.append(rmse)  # for plot
    #     self.iter_mae.append(mae)
    #     return rmse, mae, ndcg
    #
    # # test all users in test set
    # def predict_model(self):
    #     res = []
    #     list_pre = []
    #     list_true = []
    #     u_r_pr = defaultdict(list)
    #     for ind, entry in enumerate(self.rg.testSet()):
    #         user, item, rating = entry
    #         # predict
    #         pred = self.predict(user, item)
    #         # denormalize
    #         # prediction = denormalize(prediction, self.config.min_val, self.config.max_val)
    #
    #         # pred = self.checkRatingBoundary(prediction)
    #         # add prediction in order to measure
    #         # self.dao.testData[ind].append(pred)
    #         res.append([user, item, rating, pred])
    #         list_pre.append([user, item, pred])
    #         list_true.append([user, item, rating])
    #     # print(res)
    #     rmse = Metric.RMSE(res)
    #     mae = Metric.MAE(res)
    #     # ndcg = Metric.NDCG(res)
    #     sort_predata = sorted(list_pre, key=lambda x: (x[0], x[1]))
    #     sort_truedata = sorted(list_true, key=lambda x: (x[0], x[1]))
    #     # print('sort_pre: ', sort_predata)
    #     # print('sort_true: ', sort_truedata)
    #     u_r_pr = self.merge_r_pr(sort_truedata, sort_predata)
    #     # print('u_r_pr: ', u_r_pr)
    #     ndcg = Metric.calDCG_k(u_r_pr, self.config.k)
    #     print('learning_Rate = %.5f ndcg=%.5f rmse=%.5f mae=%.5f' % \
    #           (self.config.lr, ndcg, rmse, mae))
    #     # self.iter_rmse.append(rmse)  # for plot
    #     # self.iter_mae.append(mae)
    #     return rmse, mae, ndcg
    #
    # def predict(self, user, item):
    #     if self.rg.containsUser(user) and self.rg.containsItem(item):
    #         u = self.rg.user[user]
    #         i = self.rg.item[item]
    #         predictRating = sigmoid(self.Q[i].dot(self.P[u]))
    #         # predictRating = self.Q[i].dot(self.P[u])
    #         return predictRating
    #     else:
    #         return sigmoid(self.rg.globalMean)
    #         # return self.rg.globalMean
    #
    #
    # def checkRatingBoundary(self, prediction):
    #     if prediction > self.config.max_val:
    #         return self.config.max_val
    #     elif prediction < self.config.min_val:
    #         return self.config.min_val
    #     else:
    #         return round(prediction, 3)
    #
    # def save_P(self, file_name):
    #     P_dic = defaultdict(list)
    #     with open(file_name, 'wb') as f:
    #         print(self.rg.get_train_size()[0])
    #         for i in range(self.rg.get_train_size()[0]):
    #             if i in self.rg.id2user.keys():
    #                 user = self.rg.id2user[i]
    #                 P_dic[user] = self.P[i]
    #         pickle.dump(P_dic, f)
    #
    # def save_Q(self, file_name):
    #     Q_dic = defaultdict(list)
    #     with open(file_name, 'wb') as f:
    #         print(self.rg.get_train_size()[1])
    #         for i in range(self.rg.get_train_size()[1]):
    #             if i in self.rg.id2item.keys():
    #                 user = self.rg.id2item[i]
    #                 Q_dic[user] = self.Q[i]
    #         pickle.dump(Q_dic, f)
    #
    # def isConverged(self, iter):
    #     from math import isnan
    #     if isnan(self.loss):
    #         print(
    #             'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
    #         exit(-1)
    #
    #     deltaLoss = (self.lastLoss - self.loss)
    #     rmse, mae, ndcg = self.valid_model()
    #
    #     # early stopping
    #     if self.config.isEarlyStopping == True:
    #         # cond = self.lastRmse < rmse
    #         cond = self.lastNdcg > ndcg
    #         if cond:
    #             print('valid ndcg increase, so early stopping')
    #             return cond
    #         self.lastRmse = rmse
    #         self.lastMae = mae
    #         self.lastNdcg = ndcg
    #     print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f ndcg=%.5f rmse=%.5f mae=%.5f' % \
    #           (self.__class__, iter, self.loss, deltaLoss, self.config.lr, ndcg,  rmse, mae))
    #     # check if converged
    #     cond = abs(deltaLoss) < self.config.threshold
    #     converged = cond
    #     # if not converged:
    #     # 	self.updateLearningRate(iter)
    #     self.lastLoss = self.loss
    #     # shuffle(self.dao.trainingData)
    #     return converged
########################################################################################################
    # def updateLearningRate(self, iter):
    #     if iter > 1:
    #         if abs(self.lastLoss) > abs(self.loss):
    #             self.config.lr *= 1.05
    #         else:
    #             self.config.lr *= 0.5
    #     if self.config.lr > 1:
    #         self.config.lr = 1
    #
    # def show_rmse(self):
    #     '''
    #     show figure for rmse and epoch
    #     '''
    #     nums = range(len(self.iter_rmse))
    #     plt.plot(nums, self.iter_rmse, label='RMSE')
    #     plt.plot(nums, self.iter_mae, label='MAE')
    #     plt.xlabel('# of epoch')
    #     plt.ylabel('metric')
    #     plt.title(self.__class__)
    #     plt.legend()
    #     plt.show()
    #     pass

##############################################################################################
    def precision_recall_ndcg_at_k(self, k, rankedlist, test_matrix):
        idcg_k = 0
        dcg_k = 0
        n_k = k if len(test_matrix) > k else len(test_matrix)
        for i in range(n_k):
            idcg_k += 1 / math.log(i + 2, 2)

        b1 = rankedlist
        b2 = test_matrix
        s2 = set(b2)
        hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
        count = len(hits)

        for c in range(count):
            dcg_k += 1 / math.log(hits[c][0] + 2, 2)

        return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)

    def map_mrr_ndcg(self, rankedlist, test_matrix):
        ap = 0
        map = 0
        dcg = 0
        idcg = 0
        mrr = 0
        for i in range(len(test_matrix)):
            idcg += 1 / math.log(i + 2, 2)

        b1 = rankedlist
        b2 = test_matrix
        s2 = set(b2)
        hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
        count = len(hits)

        for c in range(count):
            ap += (c + 1) / (hits[c][0] + 1)
            dcg += 1 / math.log(hits[c][0] + 2, 2)

        if count != 0:
            mrr = 1 / (hits[0][0] + 1)

        if count != 0:
            map = ap / count

        return map, mrr, float(dcg / idcg)
    # 将list转为dict，只保存了list数据中的user_id 和 rating数据
    def list2dic(self, listData):
        dic = defaultdict(list)
        for i in listData:
            dic[i[0]].append(i[2])
        return dict(dic)

    # 将list转为dict，只保存了list数据中的user_id 和 item_id数据
    def list2dic_ui(self, N, listData):
        dic = defaultdict(list)
        dic_remove_less_N = {}
        cnt = {}
        for i in listData:
            if i[0] not in cnt:
                cnt[i[0]] = 0
                cnt[i[0]] += 1
            else:
                cnt[i[0]] += 1
            if cnt[i[0]] <= N:
                dic[i[0]].append(i[1])
            else:
                continue
        for u in dic:
            if len(dic[u]) == N:
                # dic_remove_less_N[u] = []
                dic_remove_less_N[u] = dic[u]
        return dict(dic_remove_less_N)

    def list2dic_ui_all(self, listData):
        dic = defaultdict(list)
        for i in listData:
                dic[i[0]].append(i[1])
        return dict(dic)

    # 将实际的rating和预测的rating合并成一个,ratingData 和 predict_ratingdata都是列表
    def merge_r_pr(self, ratingData, predict_ratingdata):
        dic = defaultdict(list)
        dic_ratingdata = self.list2dic(ratingData)
        dic_predictdata = self.list2dic(predict_ratingdata)
        key_list = list(dic_ratingdata.keys())
        for i in range(len(key_list)):
            for length in range(len(dic_ratingdata[key_list[i]])):
                dic[key_list[i]].append([dic_ratingdata[key_list[i]][length], dic_predictdata[key_list[i]][length]])
        return dic

    def dict2list(self, dict):
        dilist = []
        for eachuser_record in dict.values():
            dilist.append(eachuser_record)
        return dilist

    def dictvalues2list(self, dict):
        datadict_re = self.dict2list(dict)
        di_list = []
        for i in range(len(datadict_re)):
            di_list.append(list(datadict_re[i].items()))
        return di_list

    # DCG评价指标(所有用户的DCG均值)
    def calcDCG5(self, dist_list, rating_list, user_num):
        ratings = []
        movieid_big = []
        sum_dcg = 0.0
        for item in dist_list:
            movieid_list = []
            for i in item:
                movieid = i[0]
                movieid_list.append(movieid)
            movieid_big.append(movieid_list)
        for i in range(user_num):
            each_ratings = []
            id_list = movieid_big[i]
            ulist = rating_list[i]
            for j in range(len(id_list)):
                for k in range(len(ulist)):
                    if id_list[j] == ulist[k][0]:
                        each_ratings.append(ulist[k][1])
            ratings.append(each_ratings)
        for user_rate in ratings:
            user_dcg = 0.0
            for n in range(len(user_rate)):
                each_dcg = (pow(2, user_rate[n]) - 1) / np.log(n + 2)
                user_dcg = user_dcg + each_dcg
            sum_dcg = sum_dcg + user_dcg
        ave_dcg = sum_dcg / user_num
        return ave_dcg

    # 根据真实距离对测试集用户排序（对测试集中已有的item排序），返回一个dict
    def sortdist(self, k, dictdata):
        # dictdata = self.list2dic(data)
        sortdict = {}
        for item in dictdata.items():
            userids = item[0]
            itemids = item[1]
            sort_ratings = sorted(itemids.items(), key=lambda x: x[1], reverse=True)[:k]
            sortdict[userids] = sort_ratings
        return sortdict

    def checkRatingBoundary(self, prediction):
        if prediction > self.config.max_val:
            return self.config.max_val
        elif prediction < self.config.min_val:
            return self.config.min_val
        else:
            return round(prediction, 3)

    def valid_model(self):
        res = []
        list_pre = []
        list_true = []
        u_r_pr = defaultdict(list)
        datadict = defaultdict(dict)
        preddict = defaultdict(dict)
        pred_norm = defaultdict(dict)
        for ind, entry in enumerate(self.rg.validSet()):
            user, item, rating = entry
            # predict
            if self.config.BPR:
                pred = self.predict(user, item)
            else:
                pred = self.predict_rating(user, item)
            pred_de = denormalize(pred, self.config.min_val, self.config.max_val)
            pred_de = self.checkRatingBoundary(pred_de)
            # pred_de = pred
            res.append([user, item, rating, pred_de])
            list_pre.append([user, item, pred_de])
            list_true.append([user, item, rating])
            datadict[user][item] = rating
            preddict[user][item] = pred_de
            # pred_norm[user][item] = normalize(float(pred))
            # print(res)
        # sortdict = {}
        # # print(preddict.items())  # dict_items([(1, {2: 3, 4: 5, 6: 7}), (2, {3: 5, 4: 1, 6: 2})])
        # for item in preddict.items():
        #     userids = item[0]
        #     itemid_rating_dic = item[1]
        #     # print(itemid_rating_dic.items())  # dict_items([(2, 3), (4, 5), (6, 7)]) dict_items([(3, 5), (4, 1), (6, 2)])
        #     sort_ratings = sorted(itemid_rating_dic.items(), key=lambda x: x[1], reverse=True)[:2]
        #     sortdict[userids] = sort_ratings  # {1: [(6, 7), (4, 5)], 2: [(3, 5), (6, 2)]}
        my_ndcg = 0
        my_auc = 0
        ndcg_num = len(datadict)
        auc_num = len(datadict)
        # for u in datadict:
        #     ndcg_num, temp_ndcg = self.ndcg_a_line(datadict[u], preddict[u], ndcg_num)
        #     my_ndcg += temp_ndcg
        #     auc_num, temp_auc = self.auc_a_line(datadict[u], preddict[u], auc_num)
        #     my_auc += temp_auc
        # my_ndcg = my_ndcg / ndcg_num
        # my_auc = my_auc / auc_num
        for u in datadict:
            temp_ndcg = self.ndcg_a_line(datadict[u], preddict[u])
            my_ndcg += temp_ndcg
            temp_auc = self.auc_a_line(datadict[u], preddict[u])
            my_auc += temp_auc
        my_ndcg = my_ndcg / ndcg_num
        my_auc = my_auc / auc_num
        print('my_ndcg: ', my_ndcg, 'my_auc: ', my_auc)
        print('=' * 10)
        measure5 = Measure.rankingMeasure(datadict, preddict, 5)
        measure10 = Measure.rankingMeasure(datadict, preddict, 10)
        print(measure5)
        print(measure10)
        print('=' * 10)
        my_ndcg5 = 0
        my_ndcg10 = 0
        ndcg5_num = len(datadict)
        ndcg10_num = len(datadict)
        # for u in datadict:
        #     ndcg5_num, temp_ndcg5 = self.cal_per_user_ndcg(datadict[u], preddict[u], 5, ndcg5_num)
        #     my_ndcg5 += temp_ndcg5
        #     ndcg10_num, temp_ndcg10 = self.cal_per_user_ndcg(datadict[u], preddict[u], 10, ndcg10_num)
        #     my_ndcg10 += temp_ndcg10
        # # print(ndcg5_num)
        # # print(ndcg10_num)
        # my_ndcg5 = my_ndcg5 / ndcg5_num
        # my_ndcg10 = my_ndcg10 / ndcg10_num
        for u in datadict:
            temp_ndcg5 = self.cal_per_user_ndcg(datadict[u], preddict[u], 5)
            my_ndcg5 += temp_ndcg5
            temp_ndcg10 = self.cal_per_user_ndcg(datadict[u], preddict[u], 10)
            my_ndcg10 += temp_ndcg10
        # print(ndcg5_num)
        # print(ndcg10_num)
        my_ndcg5 = my_ndcg5 / ndcg5_num
        my_ndcg10 = my_ndcg10 / ndcg10_num
        print('my_ndcg5: ', my_ndcg5, 'my_ndcg10: ', my_ndcg10)
        print('=' * 10)
        sortdict5 = self.sortdist(5, preddict)
        dist_list5 = self.dict2list(sortdict5)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
        user_num5 = len(sortdict5.keys())
        rating_list = self.dictvalues2list(datadict)
        dcg5 = self.calcDCG5(dist_list5, rating_list, user_num5) # rating_list 关于每个user预测排序list；dist_list
        realsortdict5 = self.sortdist(5, datadict)
        realdist_list5 = self.dict2list(realsortdict5)
        maxdcg5 = self.calcDCG5(realdist_list5, rating_list, user_num5)
        ndcg5 = dcg5 / maxdcg5
        print('ndcg5, dcg5 is: ', (ndcg5, dcg5))

        sortdict10 = self.sortdist(10, preddict)
        dist_list10 = self.dict2list(sortdict10)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
        user_num10 = len(sortdict10.keys())
        # rating_list = self.dictvalues2list(datadict)
        dcg10 = self.calcDCG5(dist_list10, rating_list, user_num10)  # rating_list 关于每个user预测排序list；dist_list
        realsortdict10 = self.sortdist(10, datadict)
        realdist_list10 = self.dict2list(realsortdict10)
        maxdcg10 = self.calcDCG5(realdist_list10, rating_list, user_num10)
        ndcg10 = dcg10 / maxdcg10
        print('ndcg10, dcg10 is: ', (ndcg10, dcg10))
        print('=' * 10)
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        sort_predata = sorted(list_pre, key=lambda x: (x[0], x[1]))
        sort_truedata = sorted(list_true, key=lambda x: (x[0], x[1]))
        u_r_pr = self.merge_r_pr(sort_truedata, sort_predata)
        print(len(u_r_pr))
        fy_ndcg10 = Metric.calDCG_k(u_r_pr, 10)
        print('ndcg@10 is %f:' % fy_ndcg10)
        fy_ndcg5 = Metric.calDCG_k(u_r_pr, 5)
        print('ndcg@5 is %f:' % fy_ndcg5)
        print('=' * 10)

        sort_predata11 = sorted(list_pre, key=lambda x: (x[0], x[2]))
        sort_truedata11 = sorted(list_true, key=lambda x: (x[0], x[2]))
        truedata_ui_5 = self.list2dic_ui(5, sort_truedata11)
        predata_ui_5 = self.list2dic_ui(5, sort_predata11)
        truedata_ui_10 = self.list2dic_ui(10, sort_truedata11)
        predata_ui_10 = self.list2dic_ui(10, sort_predata11)
        # truedata_ui = self.list2dic_ui_all(sort_truedata)
        # predata_ui = self.list2dic_ui_all(sort_predata)
        # print(truedata_ui_5)
        # print(predata_ui_5)
        p_at_5 = []
        p_at_10 = []
        r_at_5 = []
        r_at_10 = []
        map = []
        mrr = []
        ndcg = []
        ndcg_at_5 = []
        ndcg_at_10 = []
        for u in truedata_ui_5:
            p_5, r_5, ndcg_5 = self.precision_recall_ndcg_at_k(5, truedata_ui_5[u], predata_ui_5[u])
            p_at_5.append(p_5)
            r_at_5.append(r_5)
            ndcg_at_5.append(ndcg_5)
        for u in truedata_ui_10:
            p_10, r_10, ndcg_10 = self.precision_recall_ndcg_at_k(10, truedata_ui_10[u], predata_ui_10[u])
            p_at_10.append(p_10)
            r_at_10.append(r_10)
            ndcg_at_10.append(ndcg_10)
            # map_u, mrr_u, ndcg_u = self.map_mrr_ndcg(truedata_ui[u], predata_ui[u])
            # map.append(map_u)
            # mrr.append(mrr_u)
            # ndcg.append(ndcg_u)
        print("rmse:" + str(rmse))
        print("mae:" + str(mae))
        if len(ndcg_at_5) != 0:
            print("ndcg@5:" + str(np.mean(ndcg_at_5)))
        if len(ndcg_at_10) != 0:
            print("ndcg@10:" + str(np.mean(ndcg_at_10)))
        if len(r_at_5) != 0:
            print("recall@5:" + str(np.mean(r_at_5)))
        if len(r_at_10) != 0:
            print("recall@10:" + str(np.mean(r_at_10)))
        if len(p_at_5) != 0:
            print("precision@5:" + str(np.mean(p_at_5)))
        if len(p_at_10) != 0:
            print("precision@10:" + str(np.mean(p_at_10)))
        return rmse, mae, my_ndcg10

    # test all users in test set
    def predict_model(self):
        res = []
        list_pre = []
        list_true = []
        u_r_pr = defaultdict(list)
        datadict = defaultdict(dict)
        preddict = defaultdict(dict)
        pred_norm = defaultdict(dict)
        for ind, entry in enumerate(self.rg.testSet()):
            user, item, rating = entry
            # predict
            if self.config.BPR:
                pred = self.predict(user, item)
            else:
                pred = self.predict_rating(user, item)
            pred_de = denormalize(pred, self.config.min_val, self.config.max_val)
            pred_de = self.checkRatingBoundary(pred_de)
            # pred_de = pred
            res.append([user, item, rating, pred_de])
            list_pre.append([user, item, pred_de])
            list_true.append([user, item, rating])
            datadict[user][item] = rating
            preddict[user][item] = pred_de
            # pred_norm[user][item] = normalize(float(pred))
        # print(res)
        # rating_list = self.dictvalues2list(datadict)
        # sortdict = {}
        # # print(preddict.items())  # dict_items([(1, {2: 3, 4: 5, 6: 7}), (2, {3: 5, 4: 1, 6: 2})])
        # for item in preddict.items():
        #     userids = item[0]
        #     itemid_rating_dic = item[1]
        #     # print(itemid_rating_dic.items())  # dict_items([(2, 3), (4, 5), (6, 7)]) dict_items([(3, 5), (4, 1), (6, 2)])
        #     sort_ratings = sorted(itemid_rating_dic.items(), key=lambda x: x[1], reverse=True)[:2]
        #     sortdict[userids] = sort_ratings  # {1: [(6, 7), (4, 5)], 2: [(3, 5), (6, 2)]}
        my_ndcg = 0
        my_auc = 0
        ndcg_num = len(datadict)
        auc_num = len(datadict)
        for u in datadict:
            temp_ndcg = self.ndcg_a_line(datadict[u], preddict[u])
            my_ndcg += temp_ndcg
            temp_auc = self.auc_a_line(datadict[u], preddict[u])
            my_auc += temp_auc
        my_ndcg = my_ndcg / ndcg_num
        my_auc = my_auc / auc_num
        print('my_ndcg: ', my_ndcg, 'my_auc: ', my_auc)
        print('=' * 10)
        measure5 = Measure.rankingMeasure(datadict, preddict, 5)
        measure10 = Measure.rankingMeasure(datadict, preddict, 10)
        print(measure5)
        print(measure10)
        print('=' * 10)
        my_ndcg5 = 0
        my_ndcg10 = 0
        ndcg5_num = len(datadict)
        ndcg10_num = len(datadict)
        # for u in datadict:
        #     ndcg5_num, temp_ndcg5 = self.cal_per_user_ndcg(datadict[u], preddict[u], 5, ndcg5_num)
        #     my_ndcg5 += temp_ndcg5
        #     ndcg10_num, temp_ndcg10 = self.cal_per_user_ndcg(datadict[u], preddict[u], 10, ndcg10_num)
        #     my_ndcg10 += temp_ndcg10
        # # print(ndcg5_num)
        # # print(ndcg10_num)
        # my_ndcg5 = my_ndcg5 / ndcg5_num
        # my_ndcg10 = my_ndcg10 / ndcg10_num
        for u in datadict:
            temp_ndcg5 = self.cal_per_user_ndcg(datadict[u], preddict[u], 5)
            my_ndcg5 += temp_ndcg5
            temp_ndcg10 = self.cal_per_user_ndcg(datadict[u], preddict[u], 10)
            my_ndcg10 += temp_ndcg10
        # print(ndcg5_num)
        # print(ndcg10_num)
        my_ndcg5 = my_ndcg5 / ndcg5_num
        my_ndcg10 = my_ndcg10 / ndcg10_num
        print('my_ndcg5: ', my_ndcg5, 'my_ndcg10: ', my_ndcg10)
        print('=' * 10)
        sortdict5 = self.sortdist(5, preddict)
        dist_list5 = self.dict2list(sortdict5)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
        user_num5 = len(sortdict5.keys())
        rating_list = self.dictvalues2list(datadict)
        dcg5 = self.calcDCG5(dist_list5, rating_list, user_num5)  # rating_list 关于每个user预测排序list；dist_list
        realsortdict5 = self.sortdist(5, datadict)
        realdist_list5 = self.dict2list(realsortdict5)
        maxdcg5 = self.calcDCG5(realdist_list5, rating_list, user_num5)
        ndcg5 = dcg5 / maxdcg5
        print('ndcg5, dcg5 is: ', (ndcg5, dcg5))

        sortdict10 = self.sortdist(10, preddict)
        dist_list10 = self.dict2list(sortdict10)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
        user_num10 = len(sortdict10.keys())
        # rating_list = self.dictvalues2list(datadict)
        dcg10 = self.calcDCG5(dist_list10, rating_list, user_num10)  # rating_list 关于每个user预测排序list；dist_list
        realsortdict10 = self.sortdist(10, datadict)
        realdist_list10 = self.dict2list(realsortdict10)
        maxdcg10 = self.calcDCG5(realdist_list10, rating_list, user_num10)
        ndcg10 = dcg10 / maxdcg10
        print('ndcg10, dcg10 is: ', (ndcg10, dcg10))
        print('='*10)
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        sort_predata = sorted(list_pre, key=lambda x: (x[0], x[1]))
        sort_truedata = sorted(list_true, key=lambda x: (x[0], x[1]))
        u_r_pr = self.merge_r_pr(sort_truedata, sort_predata)
        print(len(u_r_pr))
        fy_ndcg10 = Metric.calDCG_k(u_r_pr, 10)
        print('ndcg@10 is %f:' % fy_ndcg10)
        fy_ndcg5 = Metric.calDCG_k(u_r_pr, 5)
        print('ndcg@5 is %f:' % fy_ndcg5)
        print('=' * 10)
        sort_predata11 = sorted(list_pre, key=lambda x: (x[0], x[2]))
        sort_truedata11 = sorted(list_true, key=lambda x: (x[0], x[2]))
        truedata_ui_5 = self.list2dic_ui(5, sort_truedata11)
        predata_ui_5 = self.list2dic_ui(5, sort_predata11)
        truedata_ui_10 = self.list2dic_ui(10, sort_truedata11)
        predata_ui_10 = self.list2dic_ui(10, sort_predata11)
        p_at_5 = []
        p_at_10 = []
        r_at_5 = []
        r_at_10 = []
        map = []
        mrr = []
        ndcg = []
        ndcg_at_5 = []
        ndcg_at_10 = []
        for u in truedata_ui_5:
            p_5, r_5, ndcg_5 = self.precision_recall_ndcg_at_k(5, truedata_ui_5[u], predata_ui_5[u])
            p_at_5.append(p_5)
            r_at_5.append(r_5)
            ndcg_at_5.append(ndcg_5)
        for u in truedata_ui_10:
            p_10, r_10, ndcg_10 = self.precision_recall_ndcg_at_k(10, truedata_ui_10[u], predata_ui_10[u])
            p_at_10.append(p_10)
            r_at_10.append(r_10)
            ndcg_at_10.append(ndcg_10)
            # map_u, mrr_u, ndcg_u = self.map_mrr_ndcg(truedata_ui[u], predata_ui[u])
            # map.append(map_u)
            # mrr.append(mrr_u)
            # ndcg.append(ndcg_u)
        print("rmse:" + str(rmse))
        print("mae:" + str(mae))
        if len(ndcg_at_5) != 0:
            print("ndcg@5:" + str(np.mean(ndcg_at_5)))
        if len(ndcg_at_10) != 0:
            print("ndcg@10:" + str(np.mean(ndcg_at_10)))
        if len(r_at_5) != 0:
            print("recall@5:" + str(np.mean(r_at_5)))
        if len(r_at_10) != 0:
            print("recall@10:" + str(np.mean(r_at_10)))
        if len(p_at_5) != 0:
            print("precision@5:" + str(np.mean(p_at_5)))
        if len(p_at_10) != 0:
            print("precision@10:" + str(np.mean(p_at_10)))
        return rmse, mae, my_ndcg10

    def predict_rating(self, user, item):
        if self.rg.containsUser(user) and self.rg.containsItem(item):
            u = self.rg.user[user]
            i = self.rg.item[item]
            predictRating = self.Q[i].dot(self.P[u])
            # predictRating = self.Q[i].dot(self.P[u])
            return predictRating
        else:
            return self.rg.globalMean

    def predict(self, user, item):
        if self.rg.containsUser(user) and self.rg.containsItem(item):
            u = self.rg.user[user]
            i = self.rg.item[item]
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            # predictRating = self.Q[i].dot(self.P[u])
            return predictRating
        else:
            return sigmoid(self.rg.globalMean)

        # if self.rg.containsUser(user) and self.rg.containsItem(item):
        #     u = self.rg.user[user]
        #     i = self.rg.item[item]
        #     predictRating = sigmoid(self.Q[i].dot(self.P[u]))
        #     # predictRating = self.Q[i].dot(self.P[u])
        #     return predictRating
        # elif self.rg.containsUser(user) and not self.rg.containsItem(item):
        #     return sigmoid(self.rg.userMeans[user])
        # elif not self.rg.containsUser(user) and self.rg.containsItem(item):
        #     return sigmoid(self.rg.itemMeans[item])
        # else:
        #     return sigmoid(self.rg.globalMean)

    def isConverged(self, iter):
        from math import isnan
        if isnan(self.loss):
            print(
                'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)

        deltaLoss = (self.lastLoss - self.loss)
        # rmse, mae, ndcg = self.valid_model()
        ndcg = self.cal_valid_metric()

        # early stopping
        if self.config.isEarlyStopping == True:
            # cond = self.lastRmse < rmse
            cond = self.lastNdcg > ndcg
            if cond:
                print('valid ndcg decrease, so early stopping')
                return cond
            # self.lastRmse = rmse
            # self.lastMae = mae
            self.lastNdcg = ndcg
        # print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f ndcg=%.5f rmse=%.5f mae=%.5f' % \
        #       (self.__class__, iter, self.loss, deltaLoss, self.config.lr, ndcg,  rmse, mae))
        print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f ndcg=%.5f' % \
              (self.__class__, iter, self.loss, deltaLoss, self.config.lr, ndcg))
        # check if converged
        cond = abs(deltaLoss) < self.config.threshold
        converged = cond
        # if not converged:
        # 	self.updateLearningRate(iter)
        self.lastLoss = self.loss
        # shuffle(self.dao.trainingData)
        return converged

    def cal_per_user_ndcg(self,dictdata,dictpred,k):
        pred_ranked_item_id = []
        pred_value_list = sorted(dictpred.items(), key=lambda x: x[1], reverse=True)[:k]
        for items in pred_value_list:
            pred_ranked_item_id.append(items[0])  # 存放真实评分排序的top5
        dcg_fenzi = {}
        dcg_fenmu = {}
        dcg = {}
        for i in pred_ranked_item_id:  # 对应着去dictdata找评分
            if len(pred_ranked_item_id) == 1:
                # num -= 1
                dcg_fenzi[i] = 0 # 去掉只评论过一个item的dcg，因为永远是1.0
            else:
                dcg_fenzi[i] = 2 ** dictdata[i] - 1
        for l in range(len(pred_ranked_item_id)):
            i = l + 1
            dcg_fenmu[pred_ranked_item_id[l]] = math.log2(i + 1)
        for i in pred_ranked_item_id:
                dcg[i] = dcg_fenzi[i] / dcg_fenmu[i]
        dcg_value = 0
        for i in dcg:
            dcg_value += dcg[i]
        # print(dcg)
        # print(dcg_value)
        real_ranked_item_rating = []
        real_value_list = sorted(dictdata.items(), key=lambda x: x[1], reverse=True)[:k]
        for items in real_value_list:
            real_ranked_item_rating.append(items[1])  # 存放真实评分排序的top5
        idcg_value = 0
        rank = 1
        for i_rating in real_ranked_item_rating:
            idcg_value += (2 ** i_rating - 1) / (math.log2(rank + 1))
            # print((2 ** i_rating - 1) / (math.log2(rank + 1)))
            rank += 1
        if idcg_value <= 0: return 0
        return dcg_value / idcg_value

    def ndcg_a_line(self, y_truth, y_pred):
        result = []
        i = 0
        # for y1_i in y_truth:
        #     y2_i, i = y_pred[i], i + 1
        #     result.append((y2_i, y1_i))
        for i in y_truth:
            result.append((y_pred[i], y_truth[i]))
        sort_list = sorted(result, key=lambda d: d[0], reverse=True)
        dcg, i = 0, 0
        if len(sort_list) > 1:
            for y2_i, y1_i in sort_list:
                i += 1
                dcg += (math.pow(2, y1_i) - 1) / math.log(i + 1, 2)
        else:
            # num -= 1
            dcg = 0
        sort_list = sorted(result, key=lambda d: d[1], reverse=True)
        idcg, i = 0, 0
        for y2_i, y1_i in sort_list:
            i += 1
            idcg += (math.pow(2, y1_i) - 1) / math.log(i + 1, 2)
        if idcg <= 0: return 0
        return dcg / idcg

    def auc_a_line(self, y_truth, y_pred):
        # print(y_truth)
        # print(y_pred)
        pos_thr = 3
        result = []
        n_truth, n_false = 0, 0
        i = 0
        for i in y_truth:
            y2_i = y_pred[i]
            y1_i = y_truth[i]
            if y1_i == 0:
                n_false += 1
            else:
                n_truth += 1
            result.append((y2_i, y1_i))
        print(n_false,n_truth)
        sort_list = sorted(result, key=lambda d: d[0], reverse=True)
        idx, tot = n_false, 0
        for k, v in sort_list:
            if v >= pos_thr:
                tot += idx
            else:
                idx -= 1
        print(tot)
        if n_truth != 0 and n_false != 0:
            return float(tot) / n_truth / n_false
        else:
            # num -= 1
            return 0
    ############################################################################################
    # def predict_mat(self, user, item):
    #     predict = np.mat(user) * np.mat(item.T)
    #     return predict
    #
    # def pre_handel_valid(self, item_count):
    #     # Ensure the recommendation cannot be positive items in the training set.
    #     for u in self.rg.user_ratings.keys():
    #         for j in self.rg.user_ratings[u]:
    #             self.predict_valid[(u - 1) * item_count + j - 1] = 0
    #     # return predict
    #
    # def pre_handel_test(self, item_count):
    #     # Ensure the recommendation cannot be positive items in the training set.
    #     for u in self.rg.user_ratings.keys():
    #         for j in self.rg.user_ratings[u]:
    #             self.predict_test[(u - 1) * item_count + j - 1] = 0
    #
    # def load_test_data(self):
    #     file = open(self.config.rating_test_path, 'r')
    #     for line in file:
    #         line = line.split(',')
    #         user = int(line[0])
    #         item = int(line[1])
    #         self.test_data[user - 1][item - 1] = 1
    #
    # def load_valid_data(self):
    #     file = open(self.config.rating_valid_path, 'r')
    #     for line in file:
    #         line = line.split(',')
    #         user = int(line[0])
    #         item = int(line[1])
    #         self.valid_data[user - 1][item - 1] = 1
    #
    # def cal_test_metric(self):
    #     # user_ratings_train = self.rg.user_ratings
    #     self.load_test_data() # test_data[u][i] = 0/1
    #     item_count = self.rg.max_i
    #     user_count = self.rg.max_u
    #     for uid in range(user_count):
    #         for iid in range(item_count):
    #             if int(self.test_data[uid][iid]) != 0:
    #                 self.test[uid * item_count + iid] = 1
    #             else:
    #                 self.test[uid * item_count + iid] = 0
    #             # if iid in self.rg.id2item_test and uid in self.rg.id2user_test:
    #             #     user = self.rg.id2user_test[uid]
    #             #     item = self.rg.id2item_test[iid]
    #             #     if user in self.rg.testSet_u:
    #             #         if item in self.rg.testSet_u[user].keys():
    #             #             # self.test[uid * item_count + iid] = self.rg.testSet_u[user][item]
    #             #             self.test[uid * item_count + iid] = 1
    #             #         else:
    #             #             self.test[uid * item_count + iid] = 0
    #             #     else:
    #             #         self.test[uid * item_count + iid] = 0
    #             # else:
    #             #     self.test[uid * item_count + iid] = 0
    #     for i in range(user_count * item_count):
    #         self.test[i] = int(self.test[i])
    #
    #     # training
    #     # for i in range(self.train_count):
    #     #     self.train(user_ratings_train)
    #     print(len(np.unique(self.test)))
    #
    #     predict_matrix = self.predict_mat(self.P, self.Q)
    #     # prediction
    #     self.predict_test = predict_matrix.getA().reshape(-1)
    #     self.pre_handel_test(item_count)
    #     auc_score = roc_auc_score(self.test, self.predict_test)
    #     print('AUC:', auc_score)
    #     # Top-K evaluation
    #     return scores.topK_scores(self.test, self.predict_test, 20, user_count, item_count)
    #
    # def cal_valid_metric(self):
    #     # user_ratings_train = self.rg.user_ratings
    #     # self.load_test_data(self.test_data_path) # test_data[u][i] = 0/1
    #     self.load_valid_data()
    #     item_count = self.rg.max_i
    #     user_count = self.rg.max_u
    #     for uid in range(user_count):
    #         for iid in range(item_count):
    #             if int(self.valid_data[uid][iid]) != 0:
    #                 self.valid[uid * item_count + iid] = 1
    #             else:
    #                 self.valid[uid * item_count + iid] = 0
    #             # if iid in self.rg.id2item_valid and uid in self.rg.id2user_valid:
    #             #     user = self.rg.id2user_valid[uid]
    #             #     item = self.rg.id2item_valid[iid]
    #             #     if user in self.rg.validSet_u:
    #             #         if item in self.rg.validSet_u[user].keys():
    #             #             self.valid[uid * item_count + iid] = 1
    #             #         else:
    #             #             self.valid[uid * item_count + iid] = 0
    #             #     else:
    #             #         self.valid[uid * item_count + iid] = 0
    #             # else:
    #             #     self.valid[uid * item_count + iid] = 0
    #     for i in range(user_count * item_count):
    #         self.valid[i] = int(self.valid[i])
    #
    #     # training
    #     # for i in range(self.train_count):
    #     #     self.train(user_ratings_train)
    #     print(len(np.unique(self.valid)))
    #
    #     predict_matrix = self.predict_mat(self.P, self.Q)
    #     # prediction
    #     self.predict_valid = predict_matrix.getA().reshape(-1)
    #     self.pre_handel_valid(item_count)
    #     auc_score = roc_auc_score(self.valid, self.predict_valid)
    #     print('AUC:', auc_score)
    #     # Top-K evaluation
    #     return scores.topK_scores(self.valid, self.predict_valid, 20, user_count, item_count)

    def load_data(self):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(self.config.rating_train_path, 'r') as f:
            for line in f.readlines():
                u, i, r = line.split(",")
                u = int(u)
                i = int(i)
                r = float(r)
                if self.config.dataset_name == 'yp':
                    u = int(u) + 1
                    i = int(i) + 1
                # user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
                if r >= 3:
                    user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self):
        file = open(self.config.rating_test_path, 'r')
        for line in file:
            line = line.split(',')
            user = int(line[0])
            item = int(line[1])
            rating = float(line[2])
            if self.config.dataset_name == 'yp':
                user = int(line[0]) + 1
                item = int(line[1]) + 1
            if rating >= 3:
                self.test_data[user - 1][item - 1] = 1
            else:
                self.test_data[user - 1][item - 1] = 0

    def load_valid_data(self):
        file = open(self.config.rating_valid_path, 'r')
        for line in file:
            line = line.split(',')
            user = int(line[0])
            item = int(line[1])
            rating = float(line[2])
            if self.config.dataset_name == 'yp':
                user = int(line[0]) + 1
                item = int(line[1]) + 1
            if rating >= 3:
                self.valid_data[user - 1][item - 1] = 1
            else:
                self.valid_data[user - 1][item - 1] = 0

    def predict_mat(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def cal_test_metric(self):
        user_ratings_train = self.load_data()
        self.load_test_data()
        for u in range(self.rg.max_u):
            for item in range(self.rg.max_i):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.rg.max_i + item] = 1
                else:
                    self.test[u * self.rg.max_i + item] = 0
        for i in range(self.rg.max_u * self.rg.max_i):
            self.test[i] = int(self.test[i])
        # training
        # for i in range(self.train_count):
        #     self.train(user_ratings_train)
        predict_matrix = self.predict_mat(self.P, self.Q)
        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = self.pre_handel(user_ratings_train, self.predict_, self.rg.max_i)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.test, self.predict_, 20, self.rg.max_u, self.rg.max_i)
        return auc_score

    def cal_valid_metric(self):
        user_ratings_train = self.load_data()
        self.load_valid_data()
        for u in range(self.rg.max_u):
            for item in range(self.rg.max_i):
                if int(self.valid_data[u][item]) == 1:
                    self.valid[u * self.rg.max_i + item] = 1
                else:
                    self.valid[u * self.rg.max_i + item] = 0
        for i in range(self.rg.max_u * self.rg.max_i):
            self.valid[i] = int(self.valid[i])
        # training
        # for i in range(self.train_count):
        #     self.train(user_ratings_train)
        predict_matrix = self.predict_mat(self.P, self.Q)
        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = self.pre_handel(user_ratings_train, self.predict_, self.rg.max_i)
        auc_score = roc_auc_score(self.valid, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.valid, self.predict_, 20, self.rg.max_u, self.rg.max_i)
        return auc_score

    def pre_handel(self, set, predict, item_count):
        # Ensure the recommendation cannot be positive items in the training set.
        for u in set.keys():
            for j in set[u]:
                predict[(u - 1) * item_count + j - 1] = 0
        return predict

if __name__ == '__main__':
    mf = MF()



    # sortdict5 = mf.sortdist(5, preddict)
    # dist_list5 = mf.dict2list(sortdict5)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
    # user_num5 = len(sortdict5.keys())
    # rating_list = mf.dictvalues2list(datadict)
    # dist_list = [15,17,4,16,12,7,19,3,20,8,10,18,9,1,2,5,14,11,6,13]
    # dictdata = {15:0,17:0,4:0,16:0,12:0,7:0,19:0,3:1,20:1,8:0,10:0,18:0,9:0,1:0,2:0,5:0,14:0,11:0,6:0,13:0}
    dictdata = {1:5, 2:3, 3:2, 4:1, 5:2,6:4,7:0}
    dictpred = {1:5,2:4,3:3,4:2,5:1,6:0,7:0} # @5则预测只取排序的前5个
    k = 5
    pred_ranked_item_id = []
    pred_value_list = sorted(dictpred.items(), key=lambda x: x[1], reverse=True)[:k]
    for items in pred_value_list:
        pred_ranked_item_id.append(items[0]) #存放真实评分排序的top5
    dcg_fenzi = {}
    dcg_fenmu = {}
    dcg = {}
    for i in pred_ranked_item_id: # 对应着去dictdata找评分
        dcg_fenzi[i] = 2 ** dictdata[i] - 1
    for l in range(len(pred_ranked_item_id)):
        i = l + 1
        dcg_fenmu[pred_ranked_item_id[l]] = math.log2(i + 1)
    for i in pred_ranked_item_id:
        dcg[i] = dcg_fenzi[i] / dcg_fenmu[i]
    dcg_value = 0
    for i in dcg:
        dcg_value += dcg[i]
    # print(dcg)
    # print(dcg_value)
    real_ranked_item_rating = []
    real_value_list = sorted(dictdata.items(), key=lambda x: x[1], reverse=True)[:5]
    for items in real_value_list:
        real_ranked_item_rating.append(items[1]) #存放真实评分排序的top5
    idcg_value = 0
    rank = 1
    for i_rating in real_ranked_item_rating:
        idcg_value += (2 ** i_rating - 1) / (math.log2(rank + 1))
        # print((2 ** i_rating - 1) / (math.log2(rank + 1)))
        rank += 1
    # print(idcg_value)
    print(dcg_value/idcg_value)

    #
    # sortdict10 = mf.sortdist(10, preddict)
    # dist_list10 = mf.dict2list(sortdict10)  # [[(6, 7), (4, 5)], [(3, 5), (6, 2)]]
    # user_num10 = len(sortdict10.keys())
    # # rating_list = mf.dictvalues2list(datadict)
    # dcg10 = mf.calcDCG5(dist_list10, rating_list, user_num10)  # rating_list 关于每个user预测排序list；dist_list
    # realsortdict10 = mf.sortdist(10, datadict)
    # realdist_list10 = mf.dict2list(realsortdict10)
    # maxdcg10 = mf.calcDCG5(realdist_list10, rating_list, user_num10)
    # ndcg10 = dcg10 / maxdcg10
    # print('ndcg10, dcg10 is: ', (ndcg10, dcg10))
    # print('=' * 10)
    # sort_predata = sorted(list_pre, key=lambda x: (x[0], x[1]))
    # sort_truedata = sorted(list_true, key=lambda x: (x[0], x[1]))
    # u_r_pr = mf.merge_r_pr(sort_truedata, sort_predata)
    # print(len(u_r_pr))
    # fy_ndcg10 = Metric.calDCG_k(u_r_pr, 10)
    # print('ndcg@10 is %f:' % fy_ndcg10)
    # fy_ndcg5 = Metric.calDCG_k(u_r_pr, 5)
    # print('ndcg@5 is %f:' % fy_ndcg5)
    # print('=' * 10)
    # sort_predata11 = sorted(list_pre, key=lambda x: (x[0], x[2]))
    # sort_truedata11 = sorted(list_true, key=lambda x: (x[0], x[2]))
    # truedata_ui_5 = mf.list2dic_ui(5, sort_truedata11)
    # predata_ui_5 = mf.list2dic_ui(5, sort_predata11)
    # truedata_ui_10 = mf.list2dic_ui(10, sort_truedata11)
    # predata_ui_10 = mf.list2dic_ui(10, sort_predata11)
    # p_at_5 = []
    # p_at_10 = []
    # r_at_5 = []
    # r_at_10 = []
    # map = []
    # mrr = []
    # ndcg = []
    # ndcg_at_5 = []
    # ndcg_at_10 = []
    # for u in truedata_ui_5:
    #     p_5, r_5, ndcg_5 = mf.precision_recall_ndcg_at_k(5, truedata_ui_5[u], predata_ui_5[u])
    #     p_at_5.append(p_5)
    #     r_at_5.append(r_5)
    #     ndcg_at_5.append(ndcg_5)
    # for u in truedata_ui_10:
    #     p_10, r_10, ndcg_10 = mf.precision_recall_ndcg_at_k(10, truedata_ui_10[u], predata_ui_10[u])
    #     p_at_10.append(p_10)
    #     r_at_10.append(r_10)
    #     ndcg_at_10.append(ndcg_10)
    #     # map_u, mrr_u, ndcg_u = mf.map_mrr_ndcg(truedata_ui[u], predata_ui[u])
    #     # map.append(map_u)
    #     # mrr.append(mrr_u)
    #     # ndcg.append(ndcg_u)
    # if len(ndcg_at_5) != 0:
    #     print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    # if len(ndcg_at_10) != 0:
    #     print("ndcg@10:" + str(np.mean(ndcg_at_10)))
    # if len(r_at_5) != 0:
    #     print("recall@5:" + str(np.mean(r_at_5)))
    # if len(r_at_10) != 0:
    #     print("recall@10:" + str(np.mean(r_at_10)))
    # if len(p_at_5) != 0:
    #     print("precision@5:" + str(np.mean(p_at_5)))
    # if len(p_at_10) != 0:
    #     print("precision@10:" + str(np.mean(p_at_10)))