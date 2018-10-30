# Подход 1: градиентный бустинг "в лоб"

import pandas
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
import skimage
import pylab
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=120, threshold=np.inf)

# Names of fields
# match_id,start_time,lobby_type,r1_hero,r1_level,r1_xp,r1_gold,r1_lh,r1_kills,r1_deaths,r1_items,r2_hero,r2_level,r2_xp,r2_gold,r2_lh,r2_kills,r2_deaths,r2_items,r3_hero,r3_level,r3_xp,r3_gold,r3_lh,r3_kills,r3_deaths,r3_items,r4_hero,r4_level,r4_xp,r4_gold,r4_lh,r4_kills,r4_deaths,r4_items,r5_hero,r5_level,r5_xp,r5_gold,r5_lh,r5_kills,r5_deaths,r5_items,d1_hero,d1_level,d1_xp,d1_gold,d1_lh,d1_kills,d1_deaths,d1_items,d2_hero,d2_level,d2_xp,d2_gold,d2_lh,d2_kills,d2_deaths,d2_items,d3_hero,d3_level,d3_xp,d3_gold,d3_lh,d3_kills,d3_deaths,d3_items,d4_hero,d4_level,d4_xp,d4_gold,d4_lh,d4_kills,d4_deaths,d4_items,d5_hero,d5_level,d5_xp,d5_gold,d5_lh,d5_kills,d5_deaths,d5_items,first_blood_time,first_blood_team,first_blood_player1,first_blood_player2,radiant_bottle_time,radiant_courier_time,radiant_flying_courier_time,radiant_tpscroll_count,radiant_boots_count,radiant_ward_observer_count,radiant_ward_sentry_count,radiant_first_ward_time,dire_bottle_time,dire_courier_time,dire_flying_courier_time,dire_tpscroll_count,dire_boots_count,dire_ward_observer_count,dire_ward_sentry_count,dire_first_ward_time,duration,radiant_win,tower_status_radiant,tower_status_dire,barracks_status_radiant,barracks_status_dire
# 0,1430198770,7,11,5,2098,1489,20,0,0,7,67,3,842,991,10,0,0,4,29,5,1909,1143,10,0,0,8,20,3,757,741,6,0,0,7,105,3,732,658,4,0,1,11,4,3,1058,996,12,0,0,6,42,4,1085,986,12,0,0,4,21,5,2052,1536,23,0,0,6,37,3,742,500,2,0,0,8,84,3,958,1003,3,1,0,9,7.0,1.0,9.0,,134.0,-80.0,244.0,2,2,2,0,35.0,103.0,-84.0,221.0,3,4,2,2,-52.0,2874,1,1796,0,51,0

train_data = pandas.read_csv('data/features.csv', index_col='match_id')
test_data = pandas.read_csv('data/features_test.csv')


def X_prepare(data):
    # remove final data if exists
    X = data.loc[:, :'dire_first_ward_time']

    # fill n/a with zeros
    X.fillna(0, inplace=True)

    # add mean values of heroes valuations and XP
    # X['r_gold_mean']=np.mean(X.loc[:,['r1_gold','r2_gold','r3_gold','r4_gold','r5_gold']], axis=1)
    # X['d_gold_mean']=np.mean(X.loc[:,['d1_gold','d2_gold','d3_gold','d4_gold','d5_gold']], axis=1)
    # X['r_xp_mean']=np.mean(X.loc[:,['r1_xp','r2_xp','r3_xp','r4_xp','r5_xp']], axis=1)
    # X['d_xp_mean']=np.mean(X.loc[:,['d1_xp','d2_xp','d3_xp','d4_xp','d5_xp']], axis=1)

    # scale all values
    X = pandas.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.values)

    # drop not important, not measured data and data added as a mean values
    # X.drop(['start_time','r1_gold','r2_gold','r3_gold','r4_gold','r5_gold','d1_gold','d2_gold','d3_gold','d4_gold','d5_gold','r1_xp','r2_xp','r3_xp','r4_xp','r5_xp','d1_xp','d2_xp','d3_xp','d4_xp','d5_xp','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], 1, inplace=True)

    return X


total = len(train_data)
count = train_data.count()
count = count[count < total]

print('Features with missed values (%s rows from %s):' % (count.size, total))
print(count)

train_data = train_data[:(total//4)]

X = X_prepare(train_data)
y = train_data.loc[:, 'radiant_win']

# important_columns = list(reversed(X.columns.values[clf.feature_importances_.argsort()[-10:]]))

# print('Features:', important_columns)

# keep only important features
# X = X.loc[:, important_columns]

kfold = KFold(shuffle=True, n_splits=5)

total_scores = []
plt.figure()
plt.show()

for i in [5,10,15,20,25,30,35,40,45,50,55,60]:
    print(i)

    mean_scores = []
    n_start_time = datetime.datetime.now()
    for train, test in kfold.split(X):
        start_time = datetime.datetime.now()

        clf = GradientBoostingClassifier(n_estimators=i, verbose=False)
        clf.fit(X.iloc[train, :], y.iloc[train])

        scores = []
        for j, y_pred in enumerate(clf.staged_decision_function(X.iloc[train,:])):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            score = roc_auc_score(y.iloc[train], y_pred)
            scores.append(score)

        staged_score = np.mean(scores)

        y_pred = clf.predict(X.iloc[test])
        score = roc_auc_score(y.iloc[test], y_pred)
        mean_scores.append([score,staged_score])
        print("staged_score:", staged_score, "score:", score, ', learning rate:',clf.learning_rate, ', time elapsed:', datetime.datetime.now() - start_time)

    mean_scores = pandas.DataFrame(mean_scores)
    plt.plot(mean_scores.loc[:,0], "b")
    plt.plot(mean_scores.loc[:,1], "g")
    plt.show()

    print('mean score:', np.mean(mean_scores.loc[:, 1]), ', mean staged score:', np.mean(mean_scores.loc[:, 0]), ', time elapsed:', datetime.datetime.now() - n_start_time)
    total_scores.append([i, np.mean(mean_scores.loc[:, 0]),np.mean(mean_scores.loc[:, 1])])

total_scores = pandas.DataFrame(total_scores)
total_scores = total_scores.set_index((0))
plt.figure()
plt.plot(total_scores.loc[:,1], "c")
plt.plot(total_scores.loc[:,2], "y")
plt.show()


X_test = X_prepare(test_data.drop(['match_id'], 1))
# X_test = X_test.loc[:, important_columns]
y_pred = clf.predict_proba(X_test)


# pred = []
# for j, y_pred in enumerate(clf.staged_decision_function(X_test)):
#     y_pred = 1.0 / (1.0 + np.exp(-y_pred))
#     pred.append(y_pred)
#
# pred = np.mean(pred, axis=1)

res = pandas.DataFrame(y_pred[:,1], test_data.loc[:, 'match_id'].values, columns=['radiant_win'])
res.index.name = 'match_id'

res.to_csv("data/result-1.txt")
