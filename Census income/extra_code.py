'''
    for i in range(1,10):
        #print(training_targets.iloc[i])
        #print(id(training_targets.iloc[i]))
        #training_targets.iloc[i]
        string1 = str(training_targets.iloc[i][0])
        string2 = '<=50K'
        #print(type(p),p)
        #print(type("<=50K"))
        print(string1)
        print(''.join(sorted(string1)).strip() == ''.join(sorted(string2)).strip())
        print(sorted(string1) == sorted(string2))
        print(sorted(string1.split()) == sorted(string2.split()))
        print(set(string1.split()) == set(string2.split()))
        print(" ")
    print(training_targets.dtypes)
    print(training_features.dtypes)



dtyps = {"age":np.int16,
     "workclass":str,
     "fnlwgt":np.int32,
     "education":str,
     "education-num":np.int16,
     "marital-status":str,
     "occupation":str,
     "relationship":str,
     "race":str,
     "sex":str,
     "capital-gain":np.int32,
     "capital-loss":np.int16,
     "hours-per-week":np.int16,
     "native-country":str,
     "salary_bracket":str}
    training = pd.read_csv("adult.data.txt", sep=",",names=cols, dtype=dtyps)
    test = pd.read_csv("adult.test.txt", sep=",", names=cols, dtype=dtyps)
'''


##one hot
import pandas as pd
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

'''
dd = {'edu' :  pd.Series(['Bachelors', 'Some-college', '11th, HS-grad',
                        'Prof-school', 'Assoc-acdm', 'Some-college', '11th, HS-grad',
                        'Bachelors','Prof-school', 'Assoc-acdm']),
            'pos' : pd.Series(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                       'Self-emp-not-inc', 'Self-emp-inc','Private', 'Never-worked',
                       'Self-emp-inc', 'Federal-gov']),
            'num' : pd.Series([111,434,343,324,2,5,232,235,9,0])
}

data = pd.DataFrame(dd)
print(data)
data2 = data.replace('Bachelors', np.nan)
data2 = data2.replace('Federal-gov', np.nan)
print("Replace:")
print(data2)
mno.matrix(data2)
plt.show()

print("NA:")
print(data2.dropna())

print(pd.get_dummies(data['edu']))

data['edu'] = pd.Categorical(data['edu'])
print(pd.get_dummies(data['edu']))

print(pd.get_dummies(data))

    print("test feature: ")
    print(test_features.head())
    print(test_features.describe())
    print(test_targets.describe())
    print(test_targets.tail(50))

    #training_features.hist()
    corrmatrix = training_features.copy()
    corrmatrix['salary_bracket'] = training_targets.values
    print(corrmatrix.corr())
    training_targets.hist()
    print(min(corrmatrix['salary_bracket']))
    plt.show()


'''
cols = ["age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary_bracket"]

dtyps = [("age",'<i8'),
         ("workclass",'|S5'),
         ("fnlwgt",'<i8'),
         ("education",'|S5'),
         ("education-num",'<i8'),
         ("marital-status",'|S5'),
         ("occupation",'|S5'),
         ("relationship",'|S5'),
         ("race",'|S5'),
         ("sex",'|S5'),
         ("capital-gain",'<i8'),
         ("capital-loss",'<i8'),
         ("hours-per-week",'<i8'),
         ("native-country"'|S5'),
         ("salary_bracket"'|S5')]

training = pd.read_csv("adult.data", sep=r'\s*,\s*', names=cols, na_values="?", engine='python')

c_cols = ["workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country"]

print("Only categorical")

xx = training[c_cols].copy()
print(xx)
dd = pd.DataFrame()
for col in training[c_cols].columns:
   dd[col] = pd.get_dummies(training[col]).values.tolist()


#print(xx)

for col in c_cols:
    del training[col]

training = training.join(dd)
print(training)

#data = pd.DataFrame(dd)


#print(data)

#print(xx['workclass'])




