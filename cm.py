import numpy as np
from sklearn import linear_model, svm, neural_network
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




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




cmat = np.array([[152,28,21,16,22,17],[48,33,39,22,17,17],[39,41,44,17,23,22],[22,21,15,6,153,14],[32,18,19,22,14,107]])
classes_L = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow']  
classes = ['class {}'.format(i) for i in classes_L]
title = 'Confusion Matrix for Logistic Regression'
save_file_name = 'logreg_cm_new.png'


plot_confusion_matrix(cmat, classes, title = title, \
		savename = save_file_name)