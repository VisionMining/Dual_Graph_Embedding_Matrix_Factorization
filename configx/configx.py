class ConfigX(object):
    """
    docstring for ConfigX

    configurate the global parameters and hyper parameters

    """

    def __init__(self):
        super(ConfigX, self).__init__()
        self.dataset_name = "db"
        self.min_val = 1.0  # 0.5 1.0
        self.max_val = 5.0  # 4.0 5.0
        self.isEarlyStopping = True
        self.BPR = False

        self.rating_train_path = "../data/%s_train.csv" % self.dataset_name
        self.rating_test_path = "../data/%s_test.csv" % self.dataset_name
        self.rating_valid_path = "../data/%s_valid.csv" % self.dataset_name

        # self.rating_train_path = "../data/%s_train_50.csv" % self.dataset_name
        # self.rating_test_path = "../data/%s_test_50.csv" % self.dataset_name
        # self.rating_valid_path = "../data/%s_valid_50.csv" % self.dataset_name

        # self.rating_train_path = "../data/test/%s_train.csv" % self.dataset_name
        # self.rating_test_path = "../data/test/%s_test.csv" % self.dataset_name
        # self.rating_valid_path = "../data/test/%s_valid.csv" % self.dataset_name

        self.trust_path = '../data/%s_filter_trust.txt' % self.dataset_name
        self.sep = ','
        self.random_state = 0
        self.size = 0.8  # 0.8 0.7 0.6 0.5 0.4

        # HyperParameter
        self.k = 10
        self.coldUserRating = 5  # 用户评分数少于5条定为cold start users 5,10
        self.hotUserRating = 30  # 397个，50 40个， 30
        self.factor = 10  # 隐含因子个数户 或者5
        self.threshold = 1e-4  # 收敛的阈值
        self.lr = 0.01  # 学习率
        self.maxIter = 100
        self.lambdaP = 0.001  # 0.02
        self.lambdaQ = 0.001  # 0.02
