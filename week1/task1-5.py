# Предобработка данных в Pandas

import numpy as np
import pandas
np.set_printoptions(linewidth=250, threshold=np.inf)

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')

# 1.
# print(data.Sex.value_counts())

# 2.
# survived = data.Survived.value_counts(normalize=True).round(4) * 100
# print(survived)

# 3.
# first_class = data.Pclass.value_counts(normalize=True) * 100
# print(first_class)

# 4.
# age = data.Age.dropna().values
# print(np.average(age))
# print(np.median(age))

# 5.
# pearson = data[['SibSp', 'Parch']]
# corr = round(pearson.corr(), 2)
#
# print(data)
# print(corr)

# 6.
# names = data.loc[data.Sex == 'female']
# names = names.Name
# names = names.str.split(', ').str[1]
# names = names.str.split('(Mrs\..*\()|(Miss\. )|(Lady. \()').str[4]
# names = names.str.split('\W').str[0]
# names = names.value_counts()
# print(names)