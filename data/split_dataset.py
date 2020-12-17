import numpy as np
import pandas as pd

f = open('ep_ratings.txt', encoding='utf-8')
users_id = []
items_id = []
ratings = []
np.random.seed(2017)

for line in f:
    u, i, r = line.strip('\r\n').split(' ')
    u = int(u)
    i = int(i)
    r = float(r)
    users_id.append(u)
    items_id.append(i)
    ratings.append(r)

# get primal data
# ===================================================
data = pd.DataFrame(
    {'user_id': pd.Series(users_id),
     'item_id': pd.Series(items_id),
     'ratings': pd.Series(ratings)}
)[['user_id', 'item_id', 'ratings']]
tp_rating = data[['user_id', 'item_id', 'ratings']]

# split data
# ===================================================
n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True
tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]

tp_train.to_csv('ep_train.csv', index=False, header=None)
tp_valid.to_csv('ep_valid.csv', index=False, header=None)
tp_test.to_csv('ep_test.csv', index=False, header=None)

# tp_train.to_csv('db_train_u_%s.csv' % s, index=False, header=None)
# tp_valid.to_csv('db_valid_u_%s.csv' % s, index=False, header=None)
# tp_test.to_csv('db_test_u_%s.csv' % s, index=False, header=None)