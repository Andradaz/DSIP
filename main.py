import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


df = pd.read_csv('DSIP\creditcardoriginal.csv')

RegionDict = {0: 'Western Europe', 1: 'Eastern Europe', 2:'Northern Europe', 3: 'Southern Europe'}

rows = len(df)

RegionArray = []

for i in range (0,rows):
    x = np.random.randint(4)
    RegionArray.append(x)

df['Region'] = RegionArray

Education = {0: 'Primary', 1:'Highschool', 2: 'Bachelor', 3: 'Master', 4: 'Doctorate'}

EducationArray = []

for i in range (0,rows):
    x = np.random.randint(4)
    EducationArray.append(x)

df['Education'] = EducationArray

#Scale Amount variable

df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis=1)

print(df.head(20))

print(df.info())

# Histogram Time

fig = px.histogram(df, x="Time",
                    title = "All Transactions in Time")

fig.show()

# Histogram Time Genuine

fig = px.histogram(df.Time[df.Class==0], x="Time",
                    title = "Safe Transactions in Time",
                    color_discrete_sequence=['#3fbe7d'])

fig.show()

# # Histogram Time Fraud

fig = px.histogram(df.Time[df.Class==1], x="Time",
                    title = "Fraud Transactions in Time",
                    color_discrete_sequence=['#a92651'])

fig.show()

# #Histogram Region

fig = px.histogram(df, x="Region")

fig.show()

# # # # Education Histogram

fig = px.histogram(df, x="Education")

fig.show()

# # # # Correlation heatmap

correlation = df.corr()

fig = px.imshow(correlation)

fig.show()

## Naive Bayes
##https://www.kaggle.com/lovedeepsaini/fraud-detection-with-naive-bayes-classifier

def split_data(df):
    y = df['Class'].values #target
    
    # remove column Class (axis =1 ) 
    X = df.drop(['Class'],axis=1).values #features

    #test_size = the size of the testing dataset, default is 0.25
    #random_state  = seed
    #makes a split so that the proportion of values in the sample produced will be the same
    # as the proportion of values provided to parameter stratify

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

    print("train-set size: ", len(y_train),
      "\ntest-set size: ", len(y_test))
    print("fraud cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test


def get_predictions(clf, X_train, y_train, X_test):
    # clasifier sent as a parameter 
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    #train-set predictions
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 
    return y_pred, y_pred_prob

def print_scores(y_test,y_pred,y_pred_prob):
    # TP FN
    # FP TN
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred))

    #TP/(TP+FN)
    print("recall score: ", recall_score(y_test,y_pred))

    #TP/(TP+FP)
    print("precision score: ", precision_score(y_test,y_pred))

    # Harmonic Mean of Precision and Recall 2/((1/Recall)+(1/Precision))
    print("f1 score: ", f1_score(y_test,y_pred))

    #TP+TN/Total
    print("accuracy score: ", accuracy_score(y_test,y_pred))

    # Compute area under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    # explanation https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
    print()


print("GaussianNB algorithm")
X_train, X_test, y_train, y_test = split_data(df)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

print("DecisionTree algorithm")
X_train, X_test, y_train, y_test = split_data(df)
y_pred, y_pred_prob = get_predictions(DecisionTreeClassifier(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

print("RandomForestClassifier algorithm")
X_train, X_test, y_train, y_test = split_data(df)
y_pred, y_pred_prob = get_predictions(RandomForestClassifier(max_depth=2, random_state=0), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

### DNN implementation

# https://towardsdatascience.com/credit-card-fraud-detection-9bc8db79b956

#Split the data
y = df['Class'] #target

# remove column Class (axis =1 ) 
X = df.drop(['Class'],axis=1).values #features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = Sequential([
    Dense(input_dim = 32, units = 16, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 20, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dense(units =1, activation = 'sigmoid'),])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 5)

score = model.evaluate(X_test, y_test)
print(score)
