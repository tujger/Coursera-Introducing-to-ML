#

import pandas
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
import skimage
import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



np.set_printoptions(linewidth=120, threshold=np.inf)

# Names of fields
# match_id,start_time,lobby_type,r1_hero,r1_level,r1_xp,r1_gold,r1_lh,r1_kills,r1_deaths,r1_items,r2_hero,r2_level,r2_xp,r2_gold,r2_lh,r2_kills,r2_deaths,r2_items,r3_hero,r3_level,r3_xp,r3_gold,r3_lh,r3_kills,r3_deaths,r3_items,r4_hero,r4_level,r4_xp,r4_gold,r4_lh,r4_kills,r4_deaths,r4_items,r5_hero,r5_level,r5_xp,r5_gold,r5_lh,r5_kills,r5_deaths,r5_items,d1_hero,d1_level,d1_xp,d1_gold,d1_lh,d1_kills,d1_deaths,d1_items,d2_hero,d2_level,d2_xp,d2_gold,d2_lh,d2_kills,d2_deaths,d2_items,d3_hero,d3_level,d3_xp,d3_gold,d3_lh,d3_kills,d3_deaths,d3_items,d4_hero,d4_level,d4_xp,d4_gold,d4_lh,d4_kills,d4_deaths,d4_items,d5_hero,d5_level,d5_xp,d5_gold,d5_lh,d5_kills,d5_deaths,d5_items,first_blood_time,first_blood_team,first_blood_player1,first_blood_player2,radiant_bottle_time,radiant_courier_time,radiant_flying_courier_time,radiant_tpscroll_count,radiant_boots_count,radiant_ward_observer_count,radiant_ward_sentry_count,radiant_first_ward_time,dire_bottle_time,dire_courier_time,dire_flying_courier_time,dire_tpscroll_count,dire_boots_count,dire_ward_observer_count,dire_ward_sentry_count,dire_first_ward_time,duration,radiant_win,tower_status_radiant,tower_status_dire,barracks_status_radiant,barracks_status_dire

train_data = pandas.read_csv("data/features.csv", index_col="match_id")
test_data = pandas.read_csv("data/features_test.csv")

def X_prepare(data):
    # remove final data if exists
    X = data.loc[:, :"dire_first_ward_time"]

    # fill n/a with zeros
    X.fillna(0, inplace=True)

    # scale all values
    X = pandas.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.values)

    return X


def X_prepare_remove_cat(data):
    # remove final data if exists
    X = data.loc[:, :"dire_first_ward_time"]

    # fill n/a with zeros
    X.fillna(0, inplace=True)

    # scale all values
    X = pandas.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.values)

    X.drop(["lobby_type","r1_hero","r2_hero","r3_hero","r4_hero","r5_hero","d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], 1, inplace=True)

    return X


total = len(train_data)
count = train_data.count()
count = count[count < total]

print("Features with missed values (%s rows):" % count.size)
print(count)

# train_data = train_data[:(total//4)]

y = train_data.loc[:, "radiant_win"]
#
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = LogisticRegression(solver="lbfgs")
kfold = KFold(n_splits=5, shuffle=True)
#
# # search for optimal C
# X = X_prepare(train_data)
# print('GridSearch started for', X.size, 'elements')
# start_time = datetime.datetime.now()
# gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
# gs.fit(X, y)
# C = gs.best_estimator_.C
#
# print("Best C for pure data: %s, best score: %s, estimated time: %s" % (C, gs.best_score_, datetime.datetime.now() - start_time))
#
# # search for optimal C after removing non-number features
# X = X_prepare_remove_cat(train_data)
# print('GridSearch started for', X.size, 'elements')
# start_time = datetime.datetime.now()
# gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
# gs.fit(X, y)
# C = gs.best_estimator_.C
#
# print("Best C for clean data: %s, best score: %s, estimated time: %s" % (C, gs.best_score_, datetime.datetime.now() - start_time))



X = X_prepare_remove_cat(train_data)
# print(X.value_counts)
unique_heroes = train_data.loc[:,'r1_hero'].unique()

print("Unique heroes:", len(unique_heroes))

bag = pandas.DataFrame(unique_heroes)

X_pick = np.zeros((train_data.shape[0], np.max(unique_heroes)))

for i, match_id in enumerate(train_data.index):
    # if(i//10000 == i/10000): print(i, "...")
    for p in range(5):
        X_pick[i, train_data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, train_data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
#
X_pick = pandas.DataFrame(X_pick)
X = pandas.concat([X, X_pick], axis=1, sort=False)

# search for optimal C after adding heroes participation
# X = X_prepare_remove_cat(train_data)
print('GridSearch started for', X.size, 'elements')
start_time = datetime.datetime.now()
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
gs.fit(X, y)
C = gs.best_estimator_.C

print("Best C for data with heroes participation: %s, best score: %s, estimated time: %s" % (C, gs.best_score_, datetime.datetime.now() - start_time))


#
#
# for train, test in kfold.split(X):
#     start_time = datetime.datetime.now()
#
#     clf.fit(X.iloc[train, :], y.iloc[train])
#
#     y_pred = clf.predict(X.iloc[test])
#     score = roc_auc_score(y.iloc[test], y_pred)
#     mean_scores.append([score])
#     print("Score:", score, ", time elapsed:", datetime.datetime.now() - start_time)
#
#
#
# X_test = X_prepare(test_data.drop(["match_id"], 1))
# y_pred = clf.predict_proba(X_test)
