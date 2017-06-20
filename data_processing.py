import json
import cv2 as cv
import numpy as np
import sys
import random
import os

def read_json(num, filename_to_read = "instances_train2014.json"):
    """ This function takes in two arguments:
            1) num, the number of json object from the dataset that we want
            2) filename_to_read, this is a default argument, the name of the dataset file

        The json objects we read are review object from the COCO Dataset. We
        want to extract the review text and star rating for each review.

        This function returns a list of tuples, with the form (review, stars)
    """
    f = open(filename_to_read, "r")
    string_data = f.read()
    data = json.loads( string_data )
    result = []

    for i in range(num):
        index = random.randint(0, 82782)
        annotation = data["annotations"][index]
        imageID = annotation['image_id']
        categoryID = annotation['category_id']
        result.append( (imageID, categoryID) )

    return result


def get_processed_data(result, path):
    """ This function takes in one argument:
            1) LoR, a list of tuples with each tuple in the form of (review, stars)

        We want to processed each review, extract features, and stores all
        information in numpy array for furthur processing.

        This function returns data splitted in train and test set.
        X are all features extracted from each review, and Y are the true labels
        with the corresponding review.
    """
    X = []
    # Y = np.zeros((len(result), 1))
    Y = []

    # get filename from image_not_btw_42_142_indices
    for imageID, categoryID in result:
        length = 6-len(str(imageID))
        filename = 'COCO_train2014_000000'+ '0'*length + str(imageID) + '.jpg'
        for file in os.listdir(path):
            if file == filename:
                # print(categoryID)
                Y.append(categoryID)
                img = cv.imread(path+filename, 0)
                img = cv.resize(img, dsize=(256, 256))
                # img = img[0:256, 0:256] # TODO: re-sample
                # print (img.shape)
                # subsampling the image
                img = img.flatten()
                # print(img.shape)
                X.append(img)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    # print(X.shape)
        # features = process_review(review)
        # features_L.append(features)
        # Y.append(stars)

    # # vectorize features to numpy array
    # v = DictVectorizer(sparse=False)
    # X = v.fit_transform(features_L)

    # split the data
    size_train = int(3 * len(result) / 4)

    # get X and Y data
    X_train = X[:size_train]
    X_test = X[size_train:]

    Y_train = Y[:size_train]
    Y_test = Y[size_train:]
    # print(Y_train.min(), Y_train.max())
    return X_train, X_test, Y_train, Y_test

# main driver function
if __name__ == '__main__':
    result = read_json(1000)
    # print ('result is', result[:10])

    path="/Users/IvyLiu/Desktop/math189-Final-Project/train2014/"
    get_processed_data(result, path)
