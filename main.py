from predict import *
from data_processing import *

if __name__ == '__main__' :

    # read and process data
    print("Reading files...")
    data = read_json(5000)

    print("\nProcess data...")
    path="/Users/IvyLiu/Desktop/math189-Final-Project/train2014/"
    X_train, X_test, Y_train, Y_test = get_processed_data(data, path)

    # training and save model
    # print("\nStart training...")
    # logreg(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	# linearSVM(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
    # MLP(X_train, y_train, X_test, y_test, bin_clf = bin_clf)

    # ask for user input and predict
    YorN = input("\nDo you want to test an image? Y/N : ")

    if YorN != "N":
        filename = input("\nGreat! Please enter the filename of image you want to test: ")
        model = input("\n Which model do you want to use? Please enter the letter: \
                        (a = logistic regression, b = multilayer perceptrons, c=linear SVM")
        if model == 'a':
            model = 'finalized_model_logReg.sav'
        if model == 'b':
            model = 'finalized_model_MLP.sav'
        if model == 'c':
            model = 'finalized_model_linearSVM.sav'
        predict(filename, model)
