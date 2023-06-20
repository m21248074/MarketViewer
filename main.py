import eel

import numpy as np
import matplotlib.pyplot as plt
import uuid

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import heatmap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import MarketVision as mv

df=0

def swap_columns(df,col1,col2):
	col_list=list(df.columns)
	x,y=col_list.index(col1),col_list.index(col2)
	col_list[y],col_list[x]=col_list[x],col_list[y]
	df=df[col_list]
	return df

@eel.expose
def getData(args):
	global df
	symbol=args['symbol']
	del args['symbol']
	df=mv.getData(symbol,**args)
	df=swap_columns(df,'High','Close')
	df.index=df.index.strftime("%Y-%m-%d\n%H:%M:%S")
	df=mv.MACD(df,12,26,9)
	df=df.dropna()
	print(df)
	result=df.reset_index().values.tolist()
	return result

@eel.expose
def pca(args):
	global df
	
	print(args)
	next=int(args['target'])
	test_size=float(args['test_size'])
	
	result=mv.getTarget(df,next);
	X=result['data']
	y=result['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
	X_train_std=mv.standardize(X_train)
	X_test_std=mv.standardize(X_test)
	pca=PCA()
	pca.fit_transform(X_train_std)
	size=pca.explained_variance_ratio_.size+1
	print(pca.explained_variance_ratio_)
	plt.bar(range(1, size), pca.explained_variance_ratio_, alpha=0.5, align='center')
	plt.step(range(1, size), np.cumsum(pca.explained_variance_ratio_), where='mid')
	plt.ylabel('Explained variance ratio')
	plt.xlabel('Principal components')
	filepath="/img/"+str(uuid.uuid4())+".png"
	plt.savefig("htdocs"+filepath)
	plt.close()
	return filepath
	
@eel.expose
def randomForest(args):
	global df
	
	print(args)
	next=int(args['target'])
	test_size=float(args['test_size'])
	
	result=mv.getTarget(df,next);
	X=result['data']
	y=result['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
	
	rf=RandomForestClassifier(n_estimators=500,random_state=1)
	rf.fit(X_train,y_train)
	importances=rf.feature_importances_
	print(np.sort(importances)[::-1])
	indices = np.argsort(importances)[::-1]
	print(indices)
	plt.title('Feature Importance')
	plt.bar(range(X_train.shape[1]),importances[indices],align='center')
	plt.xticks(range(X_train.shape[1]),indices, rotation=90)
	plt.xlim([-1, X_train.shape[1]])
	plt.tight_layout()
	filepath="/img/"+str(uuid.uuid4())+".png"
	plt.savefig("htdocs"+filepath)
	plt.close()
	return filepath

@eel.expose
def heapMap(args):
	global df
	
	if 'target' in df.columns:
		df=df.drop(columns=['target'])
	
	print(args)
	next=int(args['target'])
	test_size=float(args['test_size'])
	
	X_display=df.copy()
	X_display['Next Close']=X_display['Close'].shift(-next)
	X_display=X_display.drop(columns=['Close','Adj Close','Volume'])
	X_display=X_display.dropna()
	
	print(X_display)
	
	cols=X_display.columns.values
	cm = np.corrcoef(X_display.values.T)
	hm = heatmap(cm,row_names=cols, column_names=cols)
	filepath="/img/"+str(uuid.uuid4())+".png"
	plt.savefig("htdocs"+filepath)
	plt.close()
	return filepath

@eel.expose	
def train(args):
	global df
	
	print(args)
	next=int(args['target'])
	test_size=float(args['test_size'])
	model=args['model']
	
	result=mv.getTarget(df,next);
	X=result['data']
	y=result['target']
	
	class_le=LabelEncoder()
	y=class_le.fit_transform(y)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
	
	models={
		"Logistic Regression": LogisticRegression(penalty='l2',C=100.0,solver='lbfgs',random_state=1),
		"Decision Tree": DecisionTreeClassifier(max_depth=3,criterion='entropy',random_state=0),
		"KNN": KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski'),
		"SVM": SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0),
		"GradientBoostingClassifier": GradientBoostingClassifier(),
		"Naïve Bayes Classifier": GaussianNB(),
		"Linear Regression" : LinearRegression(),
		"RandomForestRegressor" : RandomForestRegressor(n_estimators=1000,criterion='squared_error',random_state=1,n_jobs=-1)
	}
	
	clf=models[model]
	
	if model!="Decision Tree" and model!='GradientBoostingClassifier' and model!='Linear Regression' and model!="RandomForestRegressor":
		X_train=mv.standardize(X_train)
		X_test=mv.standardize(X_test)
	
	if model!='Linear Regression' and model!="RandomForestRegressor":
		clf.fit(X_train,y_train)
		prediction=clf.predict(X_test)
		return {
			'model':model,
			'Train Score': clf.score(X_train,y_train),
			'Test Score': clf.score(X_test,y_test)
		}
	
	if 'target' in df.columns:
		df=df.drop(columns=['target'])
	
	X=df.copy()
	y=X['Next Close']=X['Close'].shift(-next)
	X=X.dropna()
	X=X.drop(columns=['Close','Adj Close','Volume','Next Close'])
	y=y.dropna()
	
	size = X.shape[0]
	split_ratio = 1-test_size
	ind_split = int(split_ratio * size)
	
	X_train=X[:ind_split]
	X_test=X[ind_split:]
	y_train=y[:ind_split]
	y_test=y[ind_split:]
	
	split_time = X.index[ind_split]
	print(split_time)
	
	clf.fit(X_train,y_train)
	
	y_train_pred = clf.predict(X_train)
	y_test_pred = clf.predict(X_test)
	
	all_pred=y_train_pred.tolist()+y_test_pred.tolist()
	
	plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='Training data')

	plt.xlabel('Predicted values')
	plt.ylabel('Residuals')
	plt.legend(loc='upper left')
	plt.hlines(y=0, xmin=0, xmax=2, lw=2, color='black')
	plt.xlim([0,2])
	plt.tight_layout()
	filepath1="/img/"+str(uuid.uuid4())+".png"
	plt.savefig("htdocs"+filepath1)
	plt.close()
	
	plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='Test data')

	plt.xlabel('Predicted values')
	plt.ylabel('Residuals')
	plt.legend(loc='upper left')
	plt.hlines(y=0, xmin=0, xmax=2, lw=2, color='black')
	plt.xlim([0,2])
	plt.tight_layout()
	filepath2="/img/"+str(uuid.uuid4())+".png"
	plt.savefig("htdocs"+filepath2)
	plt.close()
	
	return {
		'model': model,
		'MSE Train': mean_squared_error(y_train, y_train_pred),
		'MAE Test': mean_squared_error(y_test, y_test_pred),
		'R^2 Train': r2_score(y_train, y_train_pred),
		'R^2 Test': r2_score(y_test, y_test_pred),
		'img': [
			{
				'id': "TrainRPImage",
				'filepath': filepath1
			},
			{
				'id': "TestRPImage",
				'filepath': filepath2
			}
		],
		'all_pred': all_pred,
		'split_time' : split_time
	}

eel.init("htdocs")
eel.start('index.html')
