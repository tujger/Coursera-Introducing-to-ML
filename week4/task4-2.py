# Составление фондового индекса

import numpy as np
import pandas
from sklearn.decomposition import PCA

np.set_printoptions(linewidth=120, threshold=np.inf)

close_prices = pandas.read_csv('data/close_prices.csv')
# date,AXP,BA,CAT,CSCO,CVX,DD,DIS,GE,GS,HD,IBM,INTC,JNJ,JPM,KO,MCD,MMM,MRK,MSFT,NKE,PFE,PG,T,TRV,UNH,UTX,V,VZ,WMT,XOM
X = close_prices[['AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','T','TRV','UNH','UTX','V','VZ','WMT','XOM']]

dj_index = pandas.read_csv('data/djia_index.csv')
# date,^DJI


for i in range(1,11):
    pca = PCA(n_components=i)
    features = pca.fit_transform(X)
    print('Components:', i, ', dispersion:', round(np.sum(pca.explained_variance_ratio_)*100,2))

pca = PCA(n_components=10)
features = pca.fit_transform(X)
features = features[:,0]

dj = dj_index['^DJI'].values

pearson = np.corrcoef(features, dj)
print('Pearson:', pearson)

print('Max ratio in 1 component:', round(np.max(pca.components_[0]),2))
print('Position of max ratio:', pca.components_[0].argsort()[-1])
print('Max ratio company:', list(X)[pca.components_[0].argsort()[-1]])

# pca.explained_variance_ratio_
# pca.components_