from typing import Tuple

import pickle
import os
import json
from sklearn import cluster, svm, model_selection
import numpy as np
import cv2

# TODO: versions of libraries that will be used:
#  Python 3.6.10
#  numpy 1.18.3
#  scikit-learn 0.22.2.post1
#  opencv-python 4.2.0.34


def load_dataset(dataset_dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(os.listdir(dataset_dir_path))):
        class_dir_path = os.path.join(dataset_dir_path, class_dir)
        for file in os.listdir(class_dir_path):
            img_file = cv2.imread(os.path.join(class_dir_path, file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def descriptor2histogram(descriptor, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptor)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptor = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = descriptor2histogram(image_descriptor, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here
    for i, img in enumerate(x):
        x[i] = cv2.resize(x[i], (910, 512))     #16:9
    return x


def project_preparation():
    img_train, label_train = load_dataset('train_data/')
    img_train = data_processing(img_train)

    img_test, label_test = load_dataset('test_data/')
    img_test = data_processing(img_test)

    feature_detector_descriptor = cv2.AKAZE_create()
    num_of_words = 440
    features = []
    for image in img_train:
        keypoints, img_descriptor = feature_detector_descriptor.detectAndCompute(image, None)
        img_descriptor = img_descriptor[0:num_of_words]
        keypoints = keypoints[0:num_of_words]
        features.extend(img_descriptor)
    features = np.asanyarray(features)
    print(features.shape)

    kmeans = cluster.KMeans(
        n_clusters=num_of_words,
        random_state=42,
        n_jobs=4
    ).fit(features)

    pickle.dump(kmeans, open("vocab_model.p", "wb"))

    x_transformed = apply_feature_transform(img_train, feature_detector_descriptor, kmeans)
    x_transformed_test = apply_feature_transform(img_test, feature_detector_descriptor, kmeans)

    Cs = [5, 10, 20]
    gammas = [90, 100, 110]
    k = ['rbf', 'sigmoid', 'poly', 'linear']
    d = [1, 2, 3, 4]
    param_grid = {'kernel': k, 'C': Cs, 'gamma': gammas, 'degree': d}
    grid_search = model_selection.GridSearchCV(svm.SVC(), param_grid)
    grid_search.fit(x_transformed, label_train)
    print(grid_search.best_params_)
    svc = svm.SVC()
    svc.set_params(**grid_search.best_params_)
    svc.fit(x_transformed, label_train)
    pickle.dump(svc, open("clf.p", "wb"))
    print(svc.score(x_transformed, label_train))
    print(svc.score(x_transformed_test, label_test))


def project():
    np.random.seed(42)

    # TODO: fill the following values
    first_name = 'Michał'
    last_name = 'Grzebyk'

    x, y = load_dataset('./../../test_data/')
    x = data_processing(x)
    
    # TODO: create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.AKAZE_create()

    # TODO: train a vocabulary model and save it using pickle.dump function
    vocab_model = pickle.load(open(f'./vocab_model.p', 'rb'))
    x_transformed = apply_feature_transform(x, feature_detector_descriptor, vocab_model)

    # TODO: train a classifier and save it using pickle.dump function
    clf = pickle.load(open(f'./clf.p', 'rb'))
    score = clf.score(x_transformed, y)
    print(f'{first_name} {last_name} score: {score}')
    with open(f'{last_name}_{first_name}_score.json', 'w') as f:
        json.dump({'score': score}, f)


if __name__ == '__main__':
    project()
