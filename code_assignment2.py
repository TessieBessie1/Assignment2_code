import umap
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# setting random seed
tf.random.set_seed(42)
np.random.seed(42)

# loading MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_data = np.concatenate((x_train, x_test), axis = 0)
y_data = np.concatenate((y_train, y_test), axis = 0)

# making brightness values in the range (0,1) and making it 2 dimensional
x_data = x_data.astype("float32") / 255.0
x_data = x_data.reshape(x_data.shape[0], -1)

# cross-validation
values_k = [100, 200, 400, 800, 1600, 3200]
skf = StratifiedKFold(n_splits = 20, shuffle = True, random_state = 42)
results = []

# create set of the algorithms
algorithms = { 
    "t-SNE": TSNE(random_state = 42,
                  n_components = 2,
                  init = "pca",
                  learning_rate = "auto",
                  method = "barnes_hut"),
    "UMAP": umap.UMAP(random_state = 42,
                      n_components = 2,
                      min_dist = 0.1,
                      n_neighbors = 15,
                      metric = "euclidean"),
    "Eigenmaps": SpectralEmbedding(random_state = 42,
                                   n_components = 2,
                                   n_neighbors = 15, 
                                   affinity = "nearest_neighbors"),
    "PCA": PCA(random_state = 42,
               n_components = 2) }

# calculating the avarage accuracy
for alg in algorithms:
    embedding = algorithms[alg].fit_transform(x_data)

    for k in values_k:
        knn = KNeighborsClassifier(n_neighbors = k)
        evaluation = cross_val_score(estimator = knn, 
                                X = embedding, 
                                y = y_data,
                                cv = skf, 
                                scoring = "accuracy")
        avg_accuracy = evaluation.mean()
        results.append({"Algorithm": alg,
                        "k": k,
                        "Accuracy": avg_accuracy})

# creating a table
accuracy = pd.DataFrame(results)
print(accuracy)



