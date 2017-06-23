import numpy as np
from sklearn import linear_model, svm, neural_network
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

# module to load data
print('==> Start to load data...')
from data_processing import*
'''
	X_train is an m_train x n array
	y_train is a 1 x m_train array
	X_test is an m_test x n array
	y_test is a 1 x m_test array
'''
result = read_json(5000)
path = "/Users/IvyLiu/Desktop/math189-Final-Project/train2014/"
X_train, X_test, y_train, y_test = get_processed_data(result, path)
print (X_train)
print(X_train.shape)
print(y_train.shape)
print('-- Number of trainimng samples: {0:4d}'.format(len(y_train)))
print('-- Number of test samples: {0:4d}'.format(len(y_test)))

def plot_confusion_matrix(cmat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savename = 'confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    print('==> Plotting confusion matrix...')
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cmat)

    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, cmat[i, j],
                 horizontalalignment="center",
                 color="white" if cmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savename, format = 'png')
    plt.close()

def bin_clf_analysis(y, y_pred):
	'''
		Analyze result of binary classification.
		Return precision, recall, F1-score.
	'''
	m = len(y)
	count_p, count_r = 0, 0
	total_p, total_r = 0, 0
	for index in range(m):
		actual, pred = y.item(index), y_pred.item(index)
		if actual == 1:
			total_r += 1
			if pred == 1:
				count_r += 1
		if pred == 1:
			total_p += 1
			if actual == 1:
				count_p += 1
	precision = 1. * count_p / total_p
	recall = 1. * count_r / total_r
	f1 = 2. * precision * recall / (precision + recall)
	return precision, recall, f1

def training(model, modelName, X_train, y_train, X_test, y_test, \
	bin_clf = False):
	'''
		Produce the training and testing accuracy for a given model
	'''
	print('==> Start training the {} model...'.format(modelName))
	t_begin = time.time()
	model.fit(X_train, y_train)
	t_end = time.time()
	print('================')
	try:
		getattr(model, 'n_iter_')
	except AttributeError:
		print('-- Model {} is not iterative'.format(modelName))
	else:
		print('-- Actual number of iterations: {}'.format(model.n_iter_))
	print('-- Time elapsed for training: {t:4.2f} seconds'.format(t = t_end - t_begin))
	# accuracy report
	print('==> Testing the {} model...'.format(modelName))
	print('-- Training accuracy: {a:4.4f}'.format(a = model.score(X_train, y_train)))
	print('-- Testing accuracy: {a:4.4f}'.format(a = model.score(X_test, y_test)))
	if bin_clf:
		y_pred = model.predict(X_test)
		precision, recall, f1 = bin_clf_analysis(y_test, y_pred)
		print('-- Testing precision: {a:4.4f}'.format(a = precision))
		print('-- Testing recall: {a:4.4f}'.format(a = recall))
		print('-- Testing F1-score: {a:4.4f}'.format(a = f1))
	print('================')

def logreg(X_train, y_train, X_test, y_test, reg = 10, lc = False, \
	bin_clf = False, cm = True, solver='sag'):
	'''
		Produce logistic regression accuracy based on the training set and
		the test set
	'''
	print('********************************')
	print('==> Setting up model for logistic regression...')
	model = linear_model.LogisticRegression(C= 1.0 / reg, verbose = False)
	training(model, 'logistic regression', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (Logistic Regression, $\lambda = {}$)'.format(reg)
		save_file_name = 'logreg'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Logistic Regression'
		save_file_name = 'logreg_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)

	# save the model to disk
	print('==> Saving the model for logistic regression...')
	filename = 'finalized_model_logReg.sav'
	pickle.dump(model, open(filename, 'wb'))
	print('********************************')


def MLP(X_train, y_train, X_test, y_test, reg = 0.01, lc = False, \
	bin_clf = False, cm = True):
	'''
		Produce the multilayer perceptron training report based on the training set and
		the test set
	'''
	print('********************************')
	print('==> Setting up MLP model')
	model = neural_network.MLPClassifier(alpha = reg, \
		hidden_layer_sizes = (1000, 1000, 1000))
	training(model, 'MLP', X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (MLP, $\lambda = {}$, hidden = [100, 100, 100])'.format(reg)
		save_file_name = 'MLP'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for MLP'
		save_file_name = 'MLP_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)
	# save the model to disk
	print('==> Saving the model for MLP...')
	filename = 'finalized_model_MLP.sav'
	pickle.dump(model, open(filename, 'wb'))
	print('********************************')

def linearSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False, \
	bin_clf = False, cm = True):
	'''
		Produce the linear support vector machine report based on the training set and
		the test set
	'''
	print('********************************')
	print('==> Setting up model for linear support vector machine...')
	model = svm.LinearSVC(C = 1.0 / reg, verbose = 0)
	training(model, 'linear SVM', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (Linear SVM, $\lambda = {}$)'.format(reg)
		save_file_name = 'linsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Linear SVM'
		save_file_name = 'linsvm_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)

	# save the model to disk
	print('==> Saving the model for linearSVM...')
	filename = 'finalized_model_linearSVM.sav'
	pickle.dump(model, open(filename, 'wb'))
	print('********************************')

def kernelSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False, \
	bin_clf = False, cm = True):
	'''
		Produce the rbf-kernel support vector machine report based on the training set and
		the test set
	'''
	print('********************************')
	print(' ==> Setting up model for rbf-kernel support vector machine...')
	model = svm.SVC(C = 1.0 / reg, verbose = 0)
	training(model, 'rbf-kernel SVM', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	title = 'Learning Curve (rbf-kernel SVM, $\lambda = {}$)'.format(reg)
	if lc:
		save_file_name = 'rbfsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Kernel SVM'
		save_file_name = 'kersvm_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)

	# save the model to disk
	print('==> Saving the model for kernelSVM...')
	filename = 'finalized_model_kernelSVM.sav'
	pickle.dump(model, open(filename, 'wb'))
	print('********************************')
	print('********************************')

def alg_batch(X_train, y_train, X_test, y_test, bin_clf = False):
	logreg(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	linearSVM(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	kernelSVM(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	MLP(X_train, y_train, X_test, y_test, bin_clf = bin_clf)

# main driver function
if __name__ == '__main__':
	# multiclass original
	print('==> Running algorithms on multiclass data with full dimension...')
	alg_batch(X_train, y_train, X_test, y_test, bin_clf = False)
	print('============================================')
	
