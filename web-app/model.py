import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('dataset_ddos_six.csv')
#input faetures
X = df[['average_dur','stddev_dur','min_dur','max_dur','srate','drate']]

#output target
encoder = LabelEncoder()
y = df[['attack']]

#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.4)

#model
model=RandomForestClassifier(n_estimators=5)
model.fit(X_train, y_train)

# Save the model as a pickle in a file
joblib.dump(model, 'model.pkl')


#prediction
predictions=model.predict(X_test)
print(predictions)
print(accuracy_score(y_test,predictions)*100)