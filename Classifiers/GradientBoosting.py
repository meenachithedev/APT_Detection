import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

test = pd.read_csv("C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\Extracted_Features_cnn.csv")
train=pd.read_csv("C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\Extracted_Features_cnn_t.csv")

#converting panda array to numpy array
train_arr=np.array(train).astype(float)
test_arr=np.array(test).astype(float)

train_X, train_y = train_arr[:,:30], train_arr[:,30:]
test_X, test_y = test_arr[:,:30], test_arr[:,30:]

pca = PCA(22)
X_train = pca.fit_transform(train_X)
X_test = pca.transform(test_X)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

classifier = GradientBoostingClassifier(random_state=2)
y_pred=classifier.fit(X_train,train_y)

print(classification_report(test_y,classifier.predict(X_test)))

