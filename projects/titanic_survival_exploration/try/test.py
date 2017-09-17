"""
This file is to try out pandas APIs/functionality in python

#Queries used
#vs.survival_stats(data, outcomes, 'Parch', ["Sex == 'male'","Parch == 1"])
#vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age < 18"])
#vs.survival_stats(data, outcomes, 'Pclass', ["Sex == 'male'","Age >= 10","Age < 18","Pclass == 1"])
#vs.survival_stats(data, outcomes, 'Parch', ["Sex == 'male'","Pclass == 3","SibSp == 2","Parch == 1","Age >= 25","Age < 45"])
#"Sex == 'male'","Age == 0", "Embarked == C", "Cabin == 0","SibSp == 1"

"""

import numpy as np
import pandas as pd

# Load the dataset
in_file = '../titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
print(full_data.head())

outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
print(data.head())


for i in range(5):
    print( data.loc[i])
    print("Outcome",outcomes[i])



print(data.shape)
predictions = pd.Series(np.zeros(len(data.index), dtype = int))
print(predictions.shape)
data['Age']=data['Age'].fillna(-1)
data['Cabin']=data['Cabin'].fillna(-1)
badData = []

for index, passenger in data.iterrows():
    sex = passenger['Sex']
    age = passenger['Age']
    pclass = passenger['Pclass']
    if (type(sex) != str) or (sex.strip() is None) or (type(age) != float) or np.isnan(age) or (type(pclass) != int):
        tup = (outcomes[index])
        badData.append(tup)
        continue
    sex = sex.strip().lower()
    if  sex == 'female':
        predictions[index] = 1
    elif sex == 'male' and age > 0 and age < 10:
        predictions[index] = 1
    elif sex == 'male' and age >= 10 and age < 18 and pclass == 1:
        predictions[index] = 1

print("badData",badData)
print(predictions.groupby(predictions.values).count())

agedata = data.loc[data['Age'] == -1]
for index, passenger in agedata.iterrows():
    if outcomes[index] == 1 and passenger['Sex'] == 'male':
        print (passenger)
        print (outcomes[index])



"""
- **Survived**: Outcome of survival (0 = No; 1 = Yes)
- **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
- **Name**: Name of passenger
- **Sex**: Sex of the passenger
- **Age**: Age of the passenger (Some entries contain `NaN`)
- **SibSp**: Number of siblings and spouses of the passenger aboard
- **Parch**: Number of parents and children of the passenger aboard
- **Ticket**: Ticket number of the passenger
- **Fare**: Fare paid by the passenger
- **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
- **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

def predictions_3(data):
   Model with multiple features. Makes a prediction with an accuracy of at least 80%.

    for index, passenger in data.iterrows():
        sex = passenger['Sex']
        age = passenger['Age']
        pclass = passenger['Pclass']
        sibSp = passenger['SibSp']
        embarked = passenger['Embarked']
        cabin = passenger['Cabin']
        if (type(sex) != str) or (sex.strip() is None) or (type(age) != float) or (type(pclass) != int) or (type(sibSp) != int):
            continue

        sex = sex.strip().lower()
        if  sex == 'female':
            predictions[index] = 1
        elif sex == 'male' and age > 0 and age < 10:
            predictions[index] = 1
        elif sex == 'male' and age >= 10 and age < 18 and pclass == 1:
            predictions[index] = 1
        elif sex == 'male' and age >= 50 and pclass == 1 and sibSp == 2:
            predictions[index] = 1
        elif sex == 'male' and age == -1 and embarked == 'C' and cabin == -1 and sibSp == 1:
            predictions[index] = 1
        elif sex == 'male' and age == -1 and embarked == 'Q' and sibSp == 2:
            predictions[index] = 1
    return predictions

# Make the predictions
predictions = predictions_3(data)
"""
