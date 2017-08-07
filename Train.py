from config import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import pickle

# fetch data and format it
data = fetch_movielens(min_rating=0)


for key, value in data.items():
    print(key, type(value), value.shape)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
# the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

# create model
model = LightFM(learning_rate=0.05,loss='warp')
# train model
model.fit(data['train'], epochs=30, num_threads=16,item_features=data['item_features'],user_features=data['user_features'],verbose=True)
# print(precision_at_k(model, data['test'], k=5,item_features=data['item_features']).mean())
print("auc score: ",auc_score(model,data['train'],item_features=data['item_features'],user_features=data['user_features']).mean())
print("precision at k: ",precision_at_k(model,data['train'],k=5,item_features=data['item_features'],user_features=data['user_features']).mean())

pickle.dump(model, open("model/model.dat", "wb"))
pickle.dump(data, open("model/data.dat", "wb"))