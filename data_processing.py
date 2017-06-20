import json
import cv2
import numpy as np

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
        annotation = data["annotations"][i]
        imageID = annotation['image_id']
        categoryID = annotation['category_id']
        result.append( (imageID, categoryID) )

    return result


def get_processed_data(result):
    """ This function takes in one argument:
            1) LoR, a list of tuples with each tuple in the form of (review, stars)

        We want to processed each review, extract features, and stores all
        information in numpy array for furthur processing.

        This function returns data splitted in train and test set.
        X are all features extracted from each review, and Y are the true labels
        with the corresponding review.
    """
    X = []
    Y = []

    # get filename from image_not_btw_42_142_indices
    for imageID, categoryID in result:
        length = 6-len(str(imageID))
        filename = 'COCO_train2014_0'+ length* '0' + str(imageID) + '.jpg'
        Y.append(categoryID)
        img = cv2.imread(filename)
        X.append(img)


        #
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

    return X_train, X_test, Y_train, Y_test

# main driver function
if __name__ == '__main__':
    result = read_json(100)
    get_processed_data(result)
