# Подход 1: градиентный бустинг "в лоб"

import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=120, threshold=np.inf)

# Names of fields
# match_id,start_time,lobby_type,r1_hero,r1_level,r1_xp,r1_gold,r1_lh,r1_kills,r1_deaths,r1_items,r2_hero,r2_level,r2_xp,r2_gold,r2_lh,r2_kills,r2_deaths,r2_items,r3_hero,r3_level,r3_xp,r3_gold,r3_lh,r3_kills,r3_deaths,r3_items,r4_hero,r4_level,r4_xp,r4_gold,r4_lh,r4_kills,r4_deaths,r4_items,r5_hero,r5_level,r5_xp,r5_gold,r5_lh,r5_kills,r5_deaths,r5_items,d1_hero,d1_level,d1_xp,d1_gold,d1_lh,d1_kills,d1_deaths,d1_items,d2_hero,d2_level,d2_xp,d2_gold,d2_lh,d2_kills,d2_deaths,d2_items,d3_hero,d3_level,d3_xp,d3_gold,d3_lh,d3_kills,d3_deaths,d3_items,d4_hero,d4_level,d4_xp,d4_gold,d4_lh,d4_kills,d4_deaths,d4_items,d5_hero,d5_level,d5_xp,d5_gold,d5_lh,d5_kills,d5_deaths,d5_items,first_blood_time,first_blood_team,first_blood_player1,first_blood_player2,radiant_bottle_time,radiant_courier_time,radiant_flying_courier_time,radiant_tpscroll_count,radiant_boots_count,radiant_ward_observer_count,radiant_ward_sentry_count,radiant_first_ward_time,dire_bottle_time,dire_courier_time,dire_flying_courier_time,dire_tpscroll_count,dire_boots_count,dire_ward_observer_count,dire_ward_sentry_count,dire_first_ward_time,duration,radiant_win,tower_status_radiant,tower_status_dire,barracks_status_radiant,barracks_status_dire
# 0,1430198770,7,11,5,2098,1489,20,0,0,7,67,3,842,991,10,0,0,4,29,5,1909,1143,10,0,0,8,20,3,757,741,6,0,0,7,105,3,732,658,4,0,1,11,4,3,1058,996,12,0,0,6,42,4,1085,986,12,0,0,4,21,5,2052,1536,23,0,0,6,37,3,742,500,2,0,0,8,84,3,958,1003,3,1,0,9,7.0,1.0,9.0,,134.0,-80.0,244.0,2,2,2,0,35.0,103.0,-84.0,221.0,3,4,2,2,-52.0,2874,1,1796,0,51,0

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


total = len(train_data)
count = train_data.count()
count = count[count < total]

print("Features with missed values (%s rows):" % count.size)
print(count)

# train_data = train_data[:(total//4)]

X = X_prepare(train_data)
y = train_data.loc[:, "radiant_win"]

kfold = KFold(shuffle=True, n_splits=5)

total_scores = []

for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
    print("Estimators:", n)

    mean_scores = []
    n_start_time = datetime.datetime.now()

    clf = GradientBoostingClassifier(n_estimators=n, verbose=False)

    # cross-validation
    for train, test in kfold.split(X):
        start_time = datetime.datetime.now()

        clf.fit(X.iloc[train, :], y.iloc[train])

        y_pred = clf.predict(X.iloc[test])
        score = roc_auc_score(y.iloc[test], y_pred)
        mean_scores.append([score])
        print("Score:", score, ", time elapsed:", datetime.datetime.now() - start_time)

    mean_scores = pandas.DataFrame(mean_scores)

    print("Mean score:", np.mean(mean_scores.loc[:, 0]), ", time elapsed:", datetime.datetime.now() - n_start_time)
    total_scores.append([n, np.mean(mean_scores.loc[:, 0])])

total_scores = pandas.DataFrame(total_scores)
total_scores = total_scores.set_index((0))

# plot the curve of quality
plt.figure()
plt.plot(total_scores.loc[:,1], "b")
plt.show()

start_time = datetime.datetime.now()

# fitting with the best n_estimators
max_quality = total_scores.idxmax()[1]
clf = GradientBoostingClassifier(n_estimators=max_quality, verbose=False)
clf.fit(X, y)

print("Best estimators:", max_quality, ", time elapsed:", datetime.datetime.now() - start_time)

X_test = X_prepare(test_data.drop(["match_id"], 1))
y_pred = clf.predict_proba(X_test)

res = pandas.DataFrame(y_pred[:,1], test_data.loc[:, "match_id"].values, columns=["radiant_win"])
res.index.name = "match_id"

res.to_csv("data/result-1.txt")
