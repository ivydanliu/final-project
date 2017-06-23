import pickle
import cv2 as cv

def predict(filename, model):
    """ This function takes in:
        (a) model, the filename of saved model
        (b) filename,, the filename of the image that we want to classify
        This function returns the predicted categoryID of the input image using
        the model selected.
    """
    # process the input image
    X_test = []
    img = cv.imread(filename, 0)
    img = cv.resize(img, dsize=(256,256)) #dependes on model
    img = img.flatten()
    X_test.append(img)

    # load the model from disk
    loaded_model = pickle.load(open(model, 'rb'))
    pred = loaded_model.predict(X_test)
    print ("\nThe predicted category using ", filename, "is ", pred)
    return pred

# main driver function
if __name__ == '__main__':
    print('==> Predicting test image category...')
    # filename1 = input("\nGreat! Please enter the filename of image you want to test: ")
    # print (filename)
    filename1 = 'demo imgs/cow.png'
    filename2 = 'demo imgs/dog.jpg'
    filename3 = 'demo imgs/cat.jpg'
    filename4 = 'demo imgs/sheep.jpg'
    filename5 = 'demo imgs/bird.jpg'
    filename6 = 'demo imgs/horse.jpg'
    # model = 'finalized_model_MLP.sav'
    model = 'finalized_model_kernelSVM.sav'
    result1 = predict(filename1, model)
    result2 = predict(filename2, model)
    result3 = predict(filename3, model)
    result4 = predict(filename4, model)
    result5 = predict(filename5, model)
    result6 = predict(filename6, model)
    print( 'id:16 ==> bird')
    print( 'id:17 ==> cat')
    print( 'id:18 ==> dog')
    print( 'id:19 ==> horse')
    print( 'id:20 ==> sheep')
    print( 'id:21 ==> cow')
    print('============================================')
