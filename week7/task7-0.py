#

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

train_data = pandas.read_csv('data/features.csv', index_col='match_id')
test_data = pandas.read_csv('data/features_test.csv')

def X_prepare(data):
    # remove final data if exists
    X = data.loc[:, :'dire_first_ward_time']

    # fill n/a with zeros
    X.fillna(0, inplace=True)

    # add mean values of heroes valuations and XP
    X['r_gold_mean']=np.mean(X.loc[:,['r1_gold','r2_gold','r3_gold','r4_gold','r5_gold']], axis=1)
    X['d_gold_mean']=np.mean(X.loc[:,['d1_gold','d2_gold','d3_gold','d4_gold','d5_gold']], axis=1)
    X['r_gold_min']=np.min(X.loc[:,['r1_gold','r2_gold','r3_gold','r4_gold','r5_gold']], axis=1)
    X['d_gold_min']=np.min(X.loc[:,['d1_gold','d2_gold','d3_gold','d4_gold','d5_gold']], axis=1)
    X['r_xp_mean']=np.mean(X.loc[:,['r1_xp','r2_xp','r3_xp','r4_xp','r5_xp']], axis=1)
    X['d_xp_mean']=np.mean(X.loc[:,['d1_xp','d2_xp','d3_xp','d4_xp','d5_xp']], axis=1)

    # scale all values
    X = pandas.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.values)

    # drop not important, not measured data and data added as a mean values
    X.drop(['start_time','r1_gold','r2_gold','r3_gold','r4_gold','r5_gold','d1_gold','d2_gold','d3_gold','d4_gold','d5_gold','r1_xp','r2_xp','r3_xp','r4_xp','r5_xp','d1_xp','d2_xp','d3_xp','d4_xp','d5_xp','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], 1, inplace=True)

    return X


X = X_prepare(train_data)
y = train_data.loc[:, 'radiant_win']

# first, get the most important features
clf = DecisionTreeClassifier()
clf.fit(X, y)

important_columns = list(reversed(X.columns.values[clf.feature_importances_.argsort()[-10:]]))

print('Features:', important_columns)

# keep only important features
X = X.loc[:, important_columns]

# total games                    97230
# first_blood_time               77677
# first_blood_team               77677
# first_blood_player1            77677
# first_blood_player2            53243
# radiant_bottle_time            81539
# radiant_courier_time           96538
# radiant_flying_courier_time    69751
# radiant_first_ward_time        95394
# dire_bottle_time               81087
# dire_courier_time              96554
# dire_flying_courier_time       71132
# dire_first_ward_time           95404


kfold = KFold(shuffle=True, n_splits=5)

total_scores = []
for i in [10,20,30,40,50]:
    print(i)

    scores = []
    n_start_time = datetime.datetime.now()
    for train, test in kfold.split(X):
        clf = GradientBoostingClassifier(n_estimators=i, verbose=False)

        start_time = datetime.datetime.now()

        clf.fit(X.iloc[train, :], y.iloc[train])

        y_pred = clf.predict(X.iloc[test])

        score = roc_auc_score(y.iloc[test], y_pred)
        scores.append(score)
        print("score:", score, ', learning rate:',clf.learning_rate, ', time elapsed:', datetime.datetime.now() - start_time)
    print('mean score:', np.mean(scores), ', time elapsed:', datetime.datetime.now() - n_start_time)
    total_scores.append(np.mean(scores))

plt.figure()
plt.plot(total_scores)
plt.show()

X_test = X_prepare(test_data.drop(['match_id'], 1))
X_test = X_test.loc[:, important_columns]
y_pred = clf.predict_proba(X_test)

res = pandas.DataFrame(y_pred[:,1], test_data.loc[:, 'match_id'].values, columns=['radiant_win'])
res.index.name = 'match_id'

res.to_csv("data/result.txt")