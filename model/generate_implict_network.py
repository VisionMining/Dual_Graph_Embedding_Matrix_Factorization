from collections import defaultdict
from configx.configx import ConfigX

class generate_implict_network():
    def __init__(self):
        self.dataSet_u = defaultdict(dict)
        self.dataSet_i = defaultdict(dict)
        self.CUNet = defaultdict(list)
        self.CINet = defaultdict(list)
        self.config = ConfigX()
        self.CU_file = "../data/%s_CUnet.txt" % self.config.dataset_name
        self.CI_file = "../data/%s_CInet.txt" % self.config.dataset_name

    def get_data_set(self):
        with open(self.config.rating_train_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(self.config.sep)
                u = int(float(u))
                i = int(float(i))
                r = float(r)
                self.dataSet_u[u][i] = r
                self.dataSet_i[i][u] = r

    def generate_cu_net(self):
        print('Building collaborative user network...')
        f = open(self.CU_file, "w")
        itemNet = {}
        for item in self.dataSet_i:  # 外层key为item 里层key为user value为评分
            if len(self.dataSet_i[item]) > 1:  # 如果item被一个以上的user评过分
                itemNet[item] = self.dataSet_i[item]  # 把这些item的里层字典赋值给itemNet

        filteredRatings = defaultdict(list)
        for item in itemNet:
            for user in itemNet[item]:
                if itemNet[item][user] > 0:
                    filteredRatings[user].append(item)  # 若user和item有交互，把 item append给filteredRatings[user]

        self.CUNet = defaultdict(list)
        weight_dic = defaultdict(dict)
        for user1 in filteredRatings:
            s1 = set(filteredRatings[user1])  # s1是与user1有交互的物品集
            for user2 in filteredRatings:
                if user1 != user2:  # 遍历filteredRatings所有user，如果不是同一个user
                    s2 = set(filteredRatings[user2])  # s2是与user2有交互的物品集
                    weight = len(s1.intersection(s2))  # 把两个不同user的评过分的物品集的交集的物品个数作为uu网络的uu权重
                    if weight > 0:
                        self.CUNet[user1] += [user2]  # 得到{user1:[user2,......]}
                        f.writelines(str(user1) + ' ' + str(user2) + ' ' + str(weight) + '\n')
                        weight_dic[user1][user2] = weight

    def generate_ci_net(self):
        print('Building collaborative item network...')
        f = open(self.CI_file, "w")
        userNet = {}
        for user in self.dataSet_u:  # 外层key为user 里层key为item value为评分
            if len(self.dataSet_u[user]) > 1:  # 如果user对一个以上的item评过分
                userNet[user] = self.dataSet_u[user]  # 把这些item的里层字典赋值给itemNet

        filteredRatings_i = defaultdict(list)
        for user in userNet:
            for item in userNet[user]:
                if userNet[user][item] > 0:
                    filteredRatings_i[item].append(user)  # 若user和item有交互，把 user append给filteredRatings_i[item]

        self.CINet = defaultdict(list)
        weight_dic_i = defaultdict(dict)
        for item1 in filteredRatings_i:
            s1 = set(filteredRatings_i[item1])  # s1是与user1有交互的物品集
            for item2 in filteredRatings_i:
                if item1 != item2:  # 遍历filteredRatings所有user，如果不是同一个user
                    s2 = set(filteredRatings_i[item2])  # s2是与user2有交互的物品集
                    weight = len(s1.intersection(s2))  # 把两个不同user的评过分的物品集的交集的物品个数作为uu网络的uu权重
                    if weight > 0:
                        self.CINet[item1] += [item2]  # 得到{user1:[user2,......]}
                        f.writelines(str(item1) + ' ' + str(item2) + ' ' + str(weight) + '\n')
                        weight_dic_i[item1][item2] = weight

    def cal_network_mean_weight(self):
        all_u_fri = 0
        all_i_fri = 0
        edge_num = 0
        all_u_weight = 0
        self.CUNet = defaultdict(list)
        weight_dic = defaultdict(dict)
        with open(self.CU_file, 'r') as file:
            for index, line in enumerate(file):
                u, f, t = line.strip().split()[:3]
                u = int(float(u))
                f = int(float(f))
                t = int(float(t))
                weight_dic[u][f] = t
                # if u not in self.CUNet.keys():
                #     self.CUNet[u] = []
                # self.CUNet[u].append(f)
                edge_num += 1
                all_u_weight += t
        print('uu over')

        # self.CINet = defaultdict(list)
        # weight_dic_i = defaultdict(dict)
        # with open(self.CI_file, 'r') as file:
        #     for index, line in enumerate(file):
        #         i, f, t = line.strip().split()[:3]
        #         i = int(float(i))
        #         f = int(float(f))
        #         t = int(float(t))
        #         weight_dic_i[i][f] = t
        #         if i not in self.CINet.keys():
        #             self.CINet[i] = []
        #         self.CINet[i].append(f)
        #         all_i_fri += 1

        # all_u_fri = 0
        min_fri = 10 ** 8
        max_fri = 0
        min_weight = 10 ** 8
        max_weight = 0
        for u in weight_dic:
            fri_len = len(weight_dic[u])
            all_u_fri += fri_len
            if fri_len < min_fri:
                min_fri = fri_len
            if fri_len > max_fri:
                max_fri = fri_len
            for f in weight_dic[u]:
                weight = weight_dic[u][f]
                if weight < min_weight:
                    min_weight = weight
                if weight > max_weight:
                    max_weight = weight

        print(all_u_fri, len(weight_dic), min_fri, max_fri)
        print(self.config.dataset_name, 'uu mean fri num is: ', (all_u_fri/len(weight_dic)))

        print(all_u_weight, edge_num, min_weight, max_weight)
        print(self.config.dataset_name, 'uu mean weight num is: ', (all_u_weight / edge_num))

        # all_i_fri = 0
        # for i in self.CINet:
        #     all_i_fri += len(self.CINet[i])
        # print('ii mean fri num is %f' % all_i_fri / len(self.CINet))

if __name__ == '__main__':
    gen_im_net = generate_implict_network()
    gen_im_net.get_data_set()
    gen_im_net.generate_cu_net()
    gen_im_net.generate_ci_net()
    # gen_im_net.cal_network_mean_weight()
