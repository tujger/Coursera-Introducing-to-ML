# Линейная регрессия: прогноз оклада по описанию вакансии

import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack


np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/salary-train.csv')
test = pandas.read_csv('data/salary-test-mini.csv')

# FullDescription,LocationNormalized,ContractTime,SalaryNormalized

y = data['SalaryNormalized']
data = data[['FullDescription','LocationNormalized','ContractTime']]
test = test[['FullDescription','LocationNormalized','ContractTime']]

data['FullDescription'] = data['FullDescription'].str.lower()
data['LocationNormalized'] = data['LocationNormalized'].str.lower()
data['ContractTime'] = data['ContractTime'].str.lower()
data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


print('Descriptions processed:\n', data.loc[[1]])

vectorizer = TfidfVectorizer(min_df=5)
X_dict = vectorizer.fit_transform(data['FullDescription'])
feature_mapping = vectorizer.get_feature_names()

print('Dict matrix size:', X_dict.size)

data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

print('Other matrix size:', X_train_categ.size)

X = hstack([X_dict, X_train_categ])

print('Joined train matrix size:', X.size)

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X,y)

print('Trained')

test['FullDescription'] = test['FullDescription'].str.lower()
test['LocationNormalized'] = test['LocationNormalized'].str.lower()
test['ContractTime'] = test['ContractTime'].str.lower()
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

X_test_dict = vectorizer.transform(test['FullDescription'])
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_dict, X_test_categ])

print('Joined test matrix size:', X_test.size)


salaries = ridge.predict(X_test)

print('Predict:', round(salaries[0], 2), round(salaries[1], 2))
