from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from photonai.base import Hyperpipe, PipelineElement
import numpy as np

from keras.metrics import Accuracy

# you can have a simple delegate
def custom_metric(y_true, y_pred):
    return np.random.random_sample()


my_pipe = Hyperpipe('example_project',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    # and here is how to register it in photonai
                    metrics=[('custom_metric', custom_metric), 'accuracy', 'f1_score'],
                    best_config_metric=Accuracy,
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SVC', kernel='rbf')
X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)