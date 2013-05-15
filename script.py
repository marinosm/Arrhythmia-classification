from numpy import genfromtxt, isnan
from scipy.stats import nanmean
from sklearn import svm, preprocessing, grid_search
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split as ttsplit

# load csv (already removed column 14)
csv = genfromtxt(fname="/Users/user/Dropbox/ML/cwk/arrhythmia.csv",delimiter=",",missing_values="?")

# split to data from the classes they belong to
classes = csv[:,-1]
data = csv[:,:-1]

# replace unkowns with the average value for that feature
for c,col in enumerate(data.T):
     for r,val in enumerate(col):
             if isnan(val):
                     data[r,c]=nanmean(col)

# scale/normalise all the data
scaler = preprocessing.MinMaxScaler()
sdata = scaler.fit_transform(data)

# join all arrhythmia classes into one => binary classification
for idx,val in enumerate(classes):
	if val != 1:
		classes[idx]=2

# produce a new dataset with reduced dimentionality by copying the original one and performing PCA
reduced_data = PCA(5).fit_transform(sdata)

# split the datasets into train&tune - modelSelection - finalEvaluation sets
trainandtune_data, finaleval_data, trainandtune_classes, finaleval_classes = ttsplit(sdata, classes, test_size=0.20, random_state=42)
trainandtune_data, modelselect_data, trainandtune_classes, modelselect_classes = ttsplit(trainandtune_data, trainandtune_classes, test_size=0.25, random_state=42)
trainandtune_reduced, finaleval_reduced, trainandtune_reducedclasses, finaleval_reducedclasses = ttsplit(reduced_data, classes, test_size=0.20, random_state=42)
trainandtune_reduced, modelselect_reduced, trainandtune_reducedclasses, modelselect_reducedclasses = ttsplit(trainandtune_reduced, trainandtune_reducedclasses, test_size=0.25, random_state=42)

# train and tune various SVMs
def correctly_classified(actual_classes, predicted_classes):
     count=0
     for idx,val in enumerate(actual_classes):
             if predicted_classes[idx]==val:
                     count+=1
     return float(count)/len(actual_classes)

machines=[]
accuracy=[]
reduced=[]

machine_type=svm.LinearSVC() 
parameters={}
parameters['tol']=[1e-4, 1e-3]
parameters['C']=[]
for i in range(-9,9):
	parameters['C'].append(2**i)
tunable = grid_search.GridSearchCV(machine_type, parameters, cv=5)
tuned = tunable.fit(trainandtune_data,trainandtune_classes)
predicted_classes = tuned.predict(modelselect_data)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_classes))
reduced.append(False)
fpr, tpr, thresholds = roc_curve(modelselect_classes, predicted_classes, 2)
print 'LinearSVC after parameter tuning = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)

tunable = grid_search.GridSearchCV(machine_type, parameters, cv=5)
tuned = tunable.fit(trainandtune_reduced,trainandtune_reducedclasses)
predicted_classes = tuned.predict(modelselect_reduced)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_reducedclasses))
reduced.append(True)
fpr, tpr, thresholds = roc_curve(modelselect_reducedclasses, predicted_classes, 2)
print 'LinearSVC after parameter tuning (After PCA) = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)


machine_type=svm.SVC()
grid=[]
case1={}
case1['probability']=[True] #to make it possible to produce the ROC curve
case1['kernel']=['linear']
case1['tol']=[1e-4, 1e-3]
case1['C']=[]
for i in range(-9,9):
	case1['C'].append(2**i)
grid.append(case1)
case2={}
case2['probability']=[True]
case2['kernel']=['rbf']
case2['degree']=[1,2,3,4,5]
case2['tol']=[1e-4, 1e-3]
case2['C']=[]
#case2['gamma']=[]
for i in range(-9,9):
	case2['C'].append(2**i)
#	case2['gamma'].append(2**i)
grid.append(case2)
case3={}
case3['probability']=[True]
case3['kernel']=['sigmoid']
case3['degree']=[1,2,3,4,5]
case3['tol']=[1e-4, 1e-3]
case3['C']=[]
#case3['coef0']=[]
for i in range(-9,9):
	case3['C'].append(2**i)
#	case3['coef0'].append(2**i)
grid.append(case3)
case4={}
case4['probability']=[True]
case4['kernel']=['poly']
case4['degree']=[1,2,3,4,5]
case4['tol']=[1e-4, 1e-3]
case4['C']=[]
#case4['gamma']=[]
#case4['coef0']=[]
for i in range(-9,9):
	case4['C'].append(2**i)
#	case4['gamma'].append(2**i)
#	case4['coef0'].append(2**i)
grid.append(case4)
tunable = grid_search.GridSearchCV(machine_type, grid, cv=5)
tuned = tunable.fit(trainandtune_data,trainandtune_classes)
predicted_classes = tuned.predict(modelselect_data)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_classes))
reduced.append(False)
fpr, tpr, thresholds = roc_curve(modelselect_classes, predicted_classes, 2)
print 'SVC after kernel selection and parameter tuning = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)

tunable = grid_search.GridSearchCV(machine_type, grid, cv=5)
tuned = tunable.fit(trainandtune_reduced,trainandtune_reducedclasses)
predicted_classes = tuned.predict(modelselect_reduced)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_reducedclasses))
reduced.append(True)
fpr, tpr, thresholds = roc_curve(modelselect_reducedclasses, predicted_classes, 2)
print 'SVC after kernel selection and parameter tuning (After PCA) = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)

machine_type=svm.NuSVC() 
for currdict in grid:
    del currdict['C']
    currdict['nu']=[]
    for i in range(2,9):
        currdict['nu'].append(float(i)/10)
tunable = grid_search.GridSearchCV(machine_type, grid)
tuned = tunable.fit(trainandtune_data,trainandtune_classes)
predicted_classes = tuned.predict(modelselect_data)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_classes))
reduced.append(False)
fpr, tpr, thresholds = roc_curve(modelselect_classes, predicted_classes, 2)
print 'NuSVC after kernel selection and parameter tuning = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)

tunable = grid_search.GridSearchCV(machine_type, grid)
tuned = tunable.fit(trainandtune_reduced,trainandtune_reducedclasses)
predicted_classes = tuned.predict(modelselect_reduced)
machines.append(tuned)
accuracy.append(correctly_classified(predicted_classes, modelselect_reducedclasses))
reduced.append(True)
fpr, tpr, thresholds = roc_curve(modelselect_reducedclasses, predicted_classes, 2)
print 'NuSVC after kernel selection and parameter tuning (After PCA) = ', tuned
print 'accuracy = ', accuracy[-1]
print 'area under ROC = ', auc(fpr, tpr)


best = accuracy.index(max(accuracy))
if reduced[best]:
    finaleval_data = finaleval_reduced
    finaleval_classes = finaleval_reducedclasses
predicted_classes = machines[best].predict(finaleval_data)
final_score = correctly_classified(predicted_classes, finaleval_classes)
fpr, tpr, thresholds = roc_curve(finaleval_classes, predicted_classes, 2)
print '----------'
print 'most accurate machine = ', machines[best]
print 'accuracy on unseen data (PCA =', reduced[best], ') = ', final_score
print 'area under ROC = ', auc(fpr, tpr)
print '----------'
