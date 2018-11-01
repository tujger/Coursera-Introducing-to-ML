# Подход 2: логистическая регрессия

import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
import datetime
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=120, threshold=np.inf)

# Names of fields
# match_id,start_time,lobby_type,r1_hero,r1_level,r1_xp,r1_gold,r1_lh,r1_kills,r1_deaths,r1_items,r2_hero,r2_level,r2_xp,r2_gold,r2_lh,r2_kills,r2_deaths,r2_items,r3_hero,r3_level,r3_xp,r3_gold,r3_lh,r3_kills,r3_deaths,r3_items,r4_hero,r4_level,r4_xp,r4_gold,r4_lh,r4_kills,r4_deaths,r4_items,r5_hero,r5_level,r5_xp,r5_gold,r5_lh,r5_kills,r5_deaths,r5_items,d1_hero,d1_level,d1_xp,d1_gold,d1_lh,d1_kills,d1_deaths,d1_items,d2_hero,d2_level,d2_xp,d2_gold,d2_lh,d2_kills,d2_deaths,d2_items,d3_hero,d3_level,d3_xp,d3_gold,d3_lh,d3_kills,d3_deaths,d3_items,d4_hero,d4_level,d4_xp,d4_gold,d4_lh,d4_kills,d4_deaths,d4_items,d5_hero,d5_level,d5_xp,d5_gold,d5_lh,d5_kills,d5_deaths,d5_items,first_blood_time,first_blood_team,first_blood_player1,first_blood_player2,radiant_bottle_time,radiant_courier_time,radiant_flying_courier_time,radiant_tpscroll_count,radiant_boots_count,radiant_ward_observer_count,radiant_ward_sentry_count,radiant_first_ward_time,dire_bottle_time,dire_courier_time,dire_flying_courier_time,dire_tpscroll_count,dire_boots_count,dire_ward_observer_count,dire_ward_sentry_count,dire_first_ward_time,duration,radiant_win,tower_status_radiant,tower_status_dire,barracks_status_radiant,barracks_status_dire

train_data = pandas.read_csv("data/features.csv", index_col="match_id")
test_data = pandas.read_csv("data/features_test.csv")

def prepare_pure(data):
    # remove final data if exists
    X = data.loc[:, :"dire_first_ward_time"]

    # fill n/a with zeros
    X.fillna(0, inplace=True)

    # scale all values
    X = pandas.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.values)

    return X


def prepare_remove_cat(data):
    X = prepare_pure(data)

    # removing categorial features
    X.drop(["lobby_type","r1_hero","r2_hero","r3_hero","r4_hero","r5_hero","d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], 1, inplace=True)
    return X


def prepare_heroes_participation(data):
    unique_heroes = data.loc[:,'r1_hero'].unique()
    print("Total unique heroes:", len(unique_heroes))

    X = prepare_remove_cat(data)

    # collecting heroes data on each game
    X_pick = np.zeros((data.shape[0], np.max(unique_heroes)))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

    X_pick = pandas.DataFrame(X_pick)
    X = pandas.concat([X, X_pick], axis=1, sort=False)

    return X


def process_clf(X, y):
    print('GridSearch for', X.size, 'elements')
    start_time = datetime.datetime.now()
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    clf = LogisticRegression(solver="sag")
    kfold = KFold(n_splits=5, shuffle=True)
    gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
    gs.fit(X, y)
    print("Found C: %s, score: %s, estimated time: %s" % (gs.best_estimator_.C, gs.best_score_, datetime.datetime.now() - start_time))
    return gs.best_estimator_.C, gs.best_score_, gs.best_estimator_


y = train_data.loc[:, "radiant_win"]

best_clf = 0
best_score = 0
best_C = 0

# search for optimal C (pure data)
print("=== Searching with pure data")
X = prepare_pure(train_data)
C, score, clf = process_clf(X,y)
if score > best_score: best_score = score; best_clf = clf; best_C = C

# search for optimal C after removing non-number features (gradient boosting)
print("\n=== Searching after removing non-number features")
X = prepare_remove_cat(train_data)
C, score, clf = process_clf(X,y)
if score > best_score: best_score = score; best_clf = clf; best_C = C

# search for optimal C with heroes participation (logistic regression)
print("\n=== Searching with heroes participation")
X = prepare_heroes_participation(train_data)
C, score, clf = process_clf(X,y)
if score > best_score: best_score = score; best_clf = clf; best_C = C

print("\n=== Best result")
print("Best C: %s, score: %s" % (best_C, best_score))


print("\n=== Finalizing")
X_test = prepare_heroes_participation(test_data.drop(["match_id"], 1))
y_pred = best_clf.predict_proba(X_test)

res = pandas.DataFrame(y_pred[:,1], test_data.loc[:, "match_id"].values, columns=["radiant_win"])
res.index.name = "match_id"

print("Min chance for radiant: %s%%, game id: %s" % (np.round(res.min()['radiant_win']*100,2), res.idxmin()['radiant_win']))
print("Max chance for radiant: %s%%, game id: %s" % (np.round(res.max()['radiant_win']*100,2), res.idxmax()['radiant_win']))
print("Mean chance for radiant: %s%%" % np.round(res.mean()['radiant_win']*100,2))
print("Median chance for radiant: %s%%" % np.round(res.median()['radiant_win']*100,2))

res.to_csv("data/result-2.txt")
