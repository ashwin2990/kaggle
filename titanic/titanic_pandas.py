import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import pylab as p
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class LogisticClassifier_compability(LogisticRegression):
        def predict(self, X):
            return self.predict_proba(X)[:, 1][:,np.newaxis]

df=pd.read_csv('train.csv',header=0)
df['AgeFill']=df['Age']
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['Gender'] = 4
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()


for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
df['Age*Class'] = df.AgeFill * df.Pclass        

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','AgeIsNull'], axis=1)
df = df.drop(['Age'], axis=1)

train_data=df.values
pd.set_option('display.max_rows', len(df))


#test data

df_test=pd.read_csv('test.csv',header=0)
df_test['AgeFill']=df_test['Age']
df_test['Gender'] = 4
df_test['Gender'] = df_test['Sex'].map( lambda x: x[0].upper() )
df_test['Gender'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

median_ages_test = np.zeros((2,3))



for i in range(0, 2):
    for j in range(0, 3):
        median_ages_test[i,j] = df_test[(df_test['Gender'] == i) & (df_test['Pclass'] == j+1)]['Age'].dropna().median()

print "start here"
for i in range(0, 2):
    for j in range(0, 3):
    	print df_test.loc[ (df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1),'AgeFill']
        df_test.loc[ (df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1),'AgeFill'] = median_ages[i,j]

     
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass 

df_test = df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_test = df_test.drop(['Age'], axis=1)



median_fares=np.zeros((1,3))

for i in range(0, 1):
    for j in range(0, 3):
    	median_fares[i,j] = df_test[(df_test['Pclass'] == j+1)]['Fare'].dropna().median()

print median_fares    	

print df_test.loc[(df_test.Fare.isnull()),'Pclass']

for j in range(0,3):
	df_test.loc[(df_test.Fare.isnull())&(df_test.Pclass == j+1),'Fare']=median_fares[0,j]


test_data=df_test.values

#print df
pd.set_option('display.max_rows', len(df_test))

#print test_data

print "correlation coefficients"
print df.corr()
print df
print df_test


#print 'Training'
#logistic_dict={0:0.5,1:0.7,2:0.7,3:0.8,4:0.5,5:1,6:1}
#print logistic_dict

base_estimator = LogisticClassifier_compability()
forest=GradientBoostingClassifier(n_estimators=900)
forest=forest.fit(train_data[0::,2::],train_data[0::,1])


ids = df_test['PassengerId'].values
output=forest.predict(test_data[0::,1::]).astype(int)

#print output

a=zip(ids,output)
write_cols=pd.DataFrame(a,columns=['PassengerId','Survived'])
write_cols.to_csv("output.csv")
print 'Done.'










