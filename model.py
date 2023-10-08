import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

class Model:
    """
    The Model class expose convenience methods to train and predict on a dataset.
    Among other things, it handles :
    - training of one model per country.
    - training with cross validation.
    - training on full dataset + prediction on test set.
    """
    def __init__(self, model_builder, ds, seed):
        """
        model_builder : function
            A function that returns a sklearn model.
    
        ds : Dataset
            A Dataset on which optical data is already loaded.

        seed : int
            The seed to use for reproducibility.
        """
        self.model_builder = model_builder
        self.ds = ds
        self.model = None
        self.models = None
        self.seed = seed
    
    def train_on_full_dataset(self):
        model = self.model_builder(self.seed)
        model.fit(self.ds.X_train, self.ds.Y_train)
        self.model = model

    def train_on_full_dataset_one_per_country(self):
        models = []
        for country in self.ds.countries:
            # Create one submodel per country
            model = Model(self.model_builder, country, self.seed) 
            model.train_on_full_dataset()
            models.append(model)
        self.models = models

    def predict_on_test(self):
        """
        Predict an already fitted model on the test set.
        Return the predictions and the IDs of the points.
        """
        # If there are submodels, predict on each of them and concatenate the results
        if self.models is not None:
            PREDS, IDS = [], []
            for model in self.models:
                pred, ids = model.predict_on_test()
                PREDS += list(pred)
                IDS += list(ids)
            PREDS = np.array(PREDS)
            IDS = np.array(IDS)
            return PREDS, IDS
        
        # If there is a single model, predict on it
        elif self.model is not None:
            return self.model.predict(self.ds.X_test), self.ds.X_test.index
        
        else:
            print('No model trained, cannot predict.')

    def train_with_cv_one_rf(self, n_splits=5, debug_level=1):
        """
        Train a single model with cross validation.
        This methods is used for experimentation, use train_on_full_dataset for final submission.
        Returns the accuracy score, the accuracy score for class 0, the accuracy score for class 1, the IDs of the points and the predictions.
        """
        X_train = self.ds.X_train
        Y_train = self.ds.Y_train
        IDs = self.ds.ids

        splits = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        accs = []
        accs_class_0 = []
        accs_class_1 = []

        idxs = []
        preds = []

        for i, (train_index, test_index) in enumerate(splits.split(X_train)):
            model = self.model_builder(self.seed)

            x_train = X_train.iloc[train_index]
            y_train = Y_train.iloc[train_index]
            x_test = X_train.iloc[test_index]
            y_test = Y_train.iloc[test_index]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            accs.append(accuracy_score(y_test, y_pred))
            accs_class_0.append(accuracy_score(y_test[y_test == 0], y_pred[y_test == 0]))
            accs_class_1.append(accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]))

            idxs += list(IDs.iloc[test_index])
            preds += list(y_pred)

        acc = np.mean(accs)
        acc_class_0 = np.mean(accs_class_0)
        acc_class_1 = np.mean(accs_class_1)

        if debug_level > 0:
            print(f'Accuracy Score : {acc:.2f}')
            print(f'Accuracy Score class 0 : {acc_class_0:.2f}')
            print(f'Accuracy Score class 1 : {acc_class_1:.2f}')

        return acc, acc_class_0, acc_class_1, idxs, preds

    def train_with_cv_one_rf_per_country(self, n_splits=3, debug_level=1, train_size=None):
        """
        Train one model per country with cross validation.
        This methods is used for experimentation, use train_on_full_dataset_one_per_country for final submission.
        Returns the accuracy score, the accuracy score for class 0, the accuracy score for class 1, the IDs of the points and the predictions.
        """

        accs = []
        accs_class_0 = []
        accs_class_1 = []
        preds = []
        idxs = []
        for country in self.ds.countries:
            if debug_level > 1:
                print(f'Training on {country.name}...')
            model = Model(self.model_builder, country, self.seed)
            
            acc, acc_class_0, acc_class_1, pred, ids = model.train_with_cv_one_rf(debug_level=0, train_size=train_size, n_splits=n_splits)
            
            preds += list(pred)
            idxs += list(ids)
            accs.append(acc)
            accs_class_0.append(acc_class_0)
            accs_class_1.append(acc_class_1)
        
        acc = np.mean(accs)
        acc_class_0 = np.mean(accs_class_0)
        acc_class_1 = np.mean(accs_class_1)

        if debug_level > 0:
            print(f'Accuracy Score : {acc:.2f}')
            print(f'Accuracy Score class 0 : {acc_class_0:.2f}')
            print(f'Accuracy Score class 1 : {acc_class_1:.2f}')

        return acc, acc_class_0, acc_class_1, idxs, preds

# Model builders
def rf_builder(seed):
    """
    A basic random forest builder (100 trees)
    """
    return RandomForestClassifier(random_state = seed)

def rf_builder_big(seed):
    """
    A wide random forest builder (500 trees)
    """
    return RandomForestClassifier(
        random_state = seed, 
        n_estimators=500, 
    )

def rf_builder_shallow(seed):
    """
    A shallow random forest builder (max_depth=10, 100 trees)
    """
    return RandomForestClassifier(
        random_state = seed, 
        n_estimators=100, 
        max_depth=10
    )

def svm_builder(seed):
    """
    A basic SVM builder
    """
    return SVC(random_state = seed)

def knn_builder_n(n):
    return KNeighborsClassifier(n_neighbors=n)

knn_builder_3 = knn_builder_n(3)
knn_builder_5 = knn_builder_n(5)
knn_builder_10 = knn_builder_n(10)