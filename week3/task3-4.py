import numpy as np
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from matplotlib import pyplot

np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/classification.csv')

pred = data['pred']
true = data['true']

tp = data[(data['true'] == 1) & (data['pred'] == 1)]
fp = data[(data['true'] == 0) & (data['pred'] == 1)]
fn = data[(data['true'] == 1) & (data['pred'] == 0)]
tn = data[(data['true'] == 0) & (data['pred'] == 0)]

true_pos = data.loc[(data['true'] == 1)]
pred_pos = data.loc[(data['pred'] == 1)]


print(len(tp), len(fp), len(fn), len(tn))

ac = round(accuracy_score(true, pred), 2)
ps = round(precision_score(true, pred), 2)
rs = round(recall_score(true, pred), 2)
f1 = round(f1_score(true, pred), 2)

print(ac, ps, rs, f1)


scores = pandas.read_csv('data/scores.csv')

# true,score_logreg,score_svm,score_knn,score_tree

s1 = round(roc_auc_score(scores['true'], scores['score_logreg']), 2)
s2 = round(roc_auc_score(scores['true'], scores['score_svm']), 2)
s3 = round(roc_auc_score(scores['true'], scores['score_knn']), 2)
s4 = round(roc_auc_score(scores['true'], scores['score_tree']), 2)

print(s1, s2, s3, s4, 'Max:', np.max([[s1,s2,s3,s4]]))

p1 = precision_recall_curve(scores['true'], scores['score_logreg'])
r1 = np.array([p1[0], p1[1]])
r1 = r1[:, r1[1,:] >= 0.7][:,-1]
# p1.loc(p1[1] > .7)

p2 = precision_recall_curve(scores['true'], scores['score_svm'])
r2 = np.array([p2[0], p2[1]])
r2 = r2[:, r2[1,:] >= 0.7][:,-1]

p3 = precision_recall_curve(scores['true'], scores['score_knn'])
r3 = np.array([p3[0], p3[1]])
r3 = r3[:, r3[1,:] >= 0.7][:,-1]

p4 = precision_recall_curve(scores['true'], scores['score_tree'])
r4 = np.array([p4[0], p4[1]])
r4 = r4[:, r4[1,:] >= 0.7][:,-1]

print(r1, r2, r3, r4)