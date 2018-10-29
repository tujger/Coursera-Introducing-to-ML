# Важность признаков


import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

np.set_printoptions(linewidth=250, threshold=np.inf)

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')
filtered = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].loc[np.isnan(data.Age) == False]

filtered.Sex[filtered.Sex == 'male'] = 1
filtered.Sex[filtered.Sex == 'female'] = 0

target = filtered.Survived

filtered = filtered[['Pclass', 'Fare', 'Age', 'Sex']]

clf = DecisionTreeClassifier(random_state=241)

clf.fit(filtered, target)

print(clf.feature_importances_)