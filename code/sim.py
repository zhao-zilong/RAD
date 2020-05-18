import numpy as np
import math

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler, \
    NearMiss, AllKNN, CondensedNearestNeighbour, \
    EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, TomekLinks  # From https://github.com/scikit-learn-contrib/imbalanced-learn

import noise
from QualitySampler import QualitySampler


class NoSampler(object):
    def fit_resample(self, X, y):
        return X, y


class Sim:

    def __init__(self, dataset, n_init, n_batch, n_samples, noise_min, noise_max, noise_type,
                 noise_steps, model_type, quality_model_type, clean_batches, sampler_type, quality_sampler_type,
                 quality_model_selection_type):
        self.n_init = n_init

        # Load testing data
        self.test_labels = np.loadtxt("data/" + dataset + "-test-labels.txt")[0:1000]
        self.test_data = np.loadtxt("data/" + dataset + "-test-data.txt")[0:1000]
        n_classes = math.floor(max(self.test_labels) + 1)
        # Load training data
        train_labels = np.loadtxt("data/" + dataset + "-train-labels.txt")
        train_data = np.loadtxt("data/" + dataset + "-train-data.txt")

        # Sim length in batches
        n_samples = min(n_samples, len(train_labels))
        self.time_horizon = math.floor((n_samples - n_init) / n_batch)

        # Randomize subset of data train data (to avoid having the same results each run
        instance_idx = np.random.choice(len(train_labels), n_samples, replace=False)
        self.train_data = train_data[instance_idx, :]
        self.train_labels = train_labels[instance_idx] # stores original labels

        # How many data we consider fully good (from where to we apply noise)
        self.last_clean = 0
        if clean_batches > 0:
            self.last_clean = n_init + (clean_batches - 1) * n_batch

        self.train_labels_noisy, self.train_labels_good, self.noise_per_batch = noise.gennoise(self.train_labels, n_classes, n_init, self.last_clean, n_batch, self.time_horizon, noise_min, noise_max, noise_type, noise_steps) # stores noisy labels

        # Maintain index arrays for the data (prediction model)
        self.all = np.arange(n_init)  # all samples received till now

        clean_boundary = min(n_init, self.last_clean)
        self.all_clean = np.arange(clean_boundary) # all clean samples received till now
        self.selected = np.arange(clean_boundary)  # clean samples selected by our quality model till now

        clean = self.train_labels_good[clean_boundary:n_init]
        clean_new = np.arange(clean_boundary, n_init)[clean == 1.0]
        self.all_clean = np.append(self.all_clean, clean_new)
        self.selected = np.append(self.selected, clean_new)

        # Maintain index arrays for the data (quality model)
        self.quality_data = np.zeros((self.train_data.shape[0]-clean_boundary, self.train_data.shape[1]+2)) # we incorporate additonal columns for the original labels and predicted labels
        self.quality_data[:,:-2] = self.train_data[clean_boundary:,:]
        self.quality_data[:,-2] = self.train_labels_noisy[clean_boundary:]
        self.quality_labels = np.ones(self.quality_data.shape[0]) # init with all good
        self.quality_labels[clean_boundary:n_init] = clean # add ground truth on noisy init data

        self.start_batch = n_init
        self.start_quality_batch = n_init - clean_boundary
        self.batch_size = n_batch

        self.random_state = np.random.randint(4000000)

        self.model_sel = self.getmodel(model_type)
        self.model_all = self.getmodel(model_type)
        self.model_init = self.getmodel(model_type)
        self.model_all_clean = self.getmodel(model_type)

        self.sampler = self.getsampler(sampler_type)

        self.quality_model = self.getmodel(quality_model_type)

        self.quality_model_selection_type = quality_model_selection_type
        self.quality_sampler = self.getsampler(quality_sampler_type)



    def getsampler(self, type):
        if type == 'none':
            sampler = NoSampler()
        elif type == 'randomunder':
            sampler = RandomUnderSampler()
        elif type == 'nearmiss':
            sampler = NearMiss()
        elif type == 'allknn':
            sampler = AllKNN()
        elif type == 'condensednn':
            sampler = CondensedNearestNeighbour()
        elif type == 'editednn':
            sampler = EditedNearestNeighbours()
        elif type == 'repeatededitednn':
            sampler = RepeatedEditedNearestNeighbours()
        elif type == 'tomeklinks':
            sampler = TomekLinks()
        elif type == 'randomover':
            sampler = RandomOverSampler()
        elif type == 'smote':
            sampler = SMOTE()
        elif type == 'adasyn':
            sampler = ADASYN()
        elif type == 'smotenc':
            sampler = SMOTENC()
        elif type == 'quality':# and self.quality_model_selection_type == 'extended':
            sampler = QualitySampler(self.n_init)
        else:
            print("Unsupported sampler %s" % type)
            exit(1)
        if type != 'none' and type != 'quality' and 'random_state' in sampler.get_params().keys():
            sampler.set_params(random_state=self.random_state)
        return sampler

    def getmodel(self, type):
        if type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif type == 'nearestcentroid':
            model = NearestCentroid()
        elif type == 'svm':
            model = SVC(gamma='scale')
        elif type == 'gaussianprocess':
            model = GaussianProcessClassifier()
        elif type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=5)
        elif type == 'ada':
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=24)
        elif type == 'mlp':
 	        model = MLPClassifier(solver='adam', hidden_layer_sizes=(28, 28), random_state=1)
        else:
            print("Unsupported quality estimator %s" % type)
            exit(1)

        if 'random_state' in model.get_params().keys():
            model.set_params(random_state=self.random_state)
        return model

    def eval_model(self, model):
        predictions = model.predict(self.test_data)
        accuracy = accuracy_score(self.test_labels, predictions)
        return accuracy

    def quality_model_fit(self):
        if self.quality_model_selection_type == 'simple':
            x_sel, y_sel = self.sampler.fit_resample(self.train_data[self.selected, :],
                                                     self.train_labels_noisy[self.selected])
            self.quality_model.fit(x_sel, y_sel)

        elif self.quality_model_selection_type == 'extended':
            if self.start_quality_batch > 0:
                X_q, y_q = self.quality_sampler.fit_resample(self.quality_data[:self.start_quality_batch, :],
                                                         self.quality_labels[:self.start_quality_batch])
                self.quality_model.fit(X_q, y_q)
        else:
            print("Unsupported quality model selection %s" % self.quality_model_selection_type)
            exit(1)

    def quality_model_select(self):
        if self.quality_model_selection_type == 'simple':
            quality_predicted_labels = self.quality_model.predict(self.train_data[self.start_batch:self.end_batch])
            same_label = np.equal(quality_predicted_labels, self.train_labels_noisy[self.start_batch:self.end_batch]) # check where we have the same label
            selected_new = np.arange(self.start_batch, self.end_batch)[same_label]
        elif self.quality_model_selection_type == 'extended' and self.start_quality_batch > 0:
            quality_predicted_labels = self.quality_model.predict(self.quality_data[self.start_quality_batch:self.end_quality_batch, :])
            selected_new = np.arange(self.start_batch, self.end_batch)[quality_predicted_labels == 1]
        else:
            selected_new = []
        return selected_new


    def run_sim(self):
        accuracy_sel = np.zeros(self.time_horizon)
        accuracy_all = np.zeros(self.time_horizon)
        accuracy_all_clean = np.zeros(self.time_horizon)

        selected_new = np.zeros(self.time_horizon)
        selected_sampled = np.zeros(self.time_horizon)
        overlaping_new = np.zeros(self.time_horizon)

        self.model_init.fit(self.train_data[np.arange(0,self.n_init), :], self.train_labels[np.arange(0,self.n_init)])
        accuracy_init = self.eval_model(self.model_init)

        #print("#seen #clean #selected")
        for batch_ix in range(self.time_horizon):
            #print("%d %d %d" % (len(self.all), len(self.all_clean), len(self.selected)))
            # Train model with current data and evaluate accuracy

            x_sel, y_sel = self.sampler.fit_resample(self.train_data[self.selected, :], self.train_labels_noisy[self.selected])
            self.model_sel.fit(x_sel, y_sel)
            accuracy_sel[batch_ix] = self.eval_model(self.model_sel)
            selected_sampled[batch_ix] = len(y_sel)

            x_all, y_all = self.sampler.fit_resample(self.train_data[self.all, :], self.train_labels_noisy[self.all])
            self.model_all.fit(x_all, y_all)
            accuracy_all[batch_ix] = self.eval_model(self.model_all)

            x_all_clean, y_all_clean = self.sampler.fit_resample(self.train_data[self.all_clean, :], self.train_labels_noisy[self.all_clean])
            self.model_all_clean.fit(x_all_clean, y_all_clean)
            accuracy_all_clean[batch_ix] = self.eval_model(self.model_all_clean)

            self.quality_model_fit()

            # Process batch
            selected_new[batch_ix], overlaping_new[batch_ix] = self.process_next_batch()

        return accuracy_init, accuracy_sel, accuracy_all, accuracy_all_clean, self.batch_size - self.noise_per_batch, selected_new, overlaping_new, selected_sampled


    def process_next_batch(self):
        self.end_batch = self.start_batch + self.batch_size
        self.end_quality_batch = self.start_quality_batch + self.batch_size

        # Classify samples by predicted versus original label and generate training data for quality model
        train_data_scope = 'batch'
        if train_data_scope == 'batch':
            predicted_labels = self.model_sel.predict(self.train_data[self.start_batch:self.end_batch])
            self.quality_data[self.start_quality_batch:self.end_quality_batch, -1] = predicted_labels

            for i in np.arange(len(predicted_labels)):
                if self.quality_data[i,-2] != self.quality_data[i,-1]:
                    self.quality_labels[i] = 0.0 # by default the value is 1
                else:
                    self.quality_labels[i] = 1.0

        elif train_data_scope == 'all':
            s = self.start_batch - self.start_quality_batch
            predicted_labels = self.model_sel.predict(self.train_data[s:self.end_batch])
            self.quality_data[:self.end_quality_batch, -1] = predicted_labels

            for i in np.arange(len(predicted_labels)):
                if self.quality_data[i,-2] != self.quality_data[i,-1]:
                    self.quality_labels[i] = 0.0 # by default the value is 1
                else:
                    self.quality_labels[i] = 1.0
        else:
            print("Unsupported train_data_scope %s" % train_data_scope)
            exit(1)


        # Use quality model to select sample units to use

        selected_new = self.quality_model_select()
        self.selected = np.append(self.selected, selected_new)

        self.all = np.append(self.all, np.arange(self.start_batch, self.end_batch))
        clean = self.train_labels_good[self.start_batch:self.end_batch]
        clean_new = np.arange(self.start_batch, self.end_batch)[clean == 1.0]
        self.all_clean = np.append(self.all_clean, clean_new)

        # compute how many good trues we have selected
        overlapping_new = set(clean_new).intersection(selected_new)

        # move to next batch
        self.start_batch = self.end_batch
        self.start_quality_batch = self.end_quality_batch

        return len(selected_new), len(overlapping_new)
