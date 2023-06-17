#Importing Machine Learning Libraries
import deepchem as dc
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


#Plotting and Data
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

#Other Libraries 
import math
import copy
import pickle 
import os
import itertools
import collections
from typing import List, Optional, Union, Tuple, Dict, Type, Any, Callable, Generator


#Custom Type Aliases
dc_dataset = Type[dc.data.datasets.Dataset]
dc_model = Type[Union[dc.models.KerasModel,dc.models.SklearnModel]]
dc_splitter = Type[dc.splits.Splitter]


#Hyper-parameter ranges
LOGISTIC_PARAMS = {
     'C': [.001,.01,.1,.25,.5,.75,1],
}

RFC_PARAMS = {
    'max_features': [40,60,100],
    'n_estimators': [500,1000,2000]
}

XGB_PARAMS = {
    'eta': [.001,.01,.1,.5,1],
    'reg_lambda': [.00001,.0001,.001,.01,.1],
    'subsample': [.5,.6,.7,.8,.9,1]
}

NN_PARAMS ={
    'dropout': [.2,.3,.4,.5,.6,.7,.8],
    'weights_penalty': [.000001,.0001,.001,.01,.1]
}

def construct_representation(raw_data: pd.DataFrame, featurizer: dc.feat) -> dc_dataset:
    """
    Function to create chemistry-informed numerical representation of data, given the raw data and a featurizer object
    https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html
    :param raw_data: A pandas dataframe containing the raw data to be featurized
    :param featurizer: A deepchem featurizer object
    :return: A deepchem dataset object
    """

    #Specifying features, and targets or tasks
    loader = dc.data.CSVLoader(tasks = ['p_np'], feature_field="smiles", featurizer= featurizer)

    #Creates transformed dataset and wraps a deepchem dataset object
    new_representation = loader.create_dataset('data/BBBP.csv')

    return new_representation


def k_fold_data_splitting(dataset: dc_dataset, splitter: dc_splitter, folds: int) -> List[dc_dataset]:
    """
    Function that splits deepchem dataset into k-folds
    :param dataset: A deepchem dataset object to be split
    :param splitter: A deepchem data splitting object
    :param folds: An int specify the number of folds to split the deepchem dataset by
    :return: A list of split deepchem datasets
    """

    return splitter.k_fold_split(dataset,folds)


# Model Building Functions (functions that take in hyperparameter and return initialized models/estimators)

def logistic_model_builder(**model_params: Dict[str, List[Union[str, int, float]]]) -> dc.models.SklearnModel:
    """
    Function that initializes a logistic regression model given model parameters
    :param model_params: A dictionary that contains model parameters for a logistic regression model
    :return: A deepchem scikit-learn model object
    """

    C = model_params['C']
    logistic_model = LogisticRegression(C = C)

    #Wrapping scikit-learn estimator in deep_chem model class
    deepchem_logistic_model = dc.models.SklearnModel(logistic_model)

    return deepchem_logistic_model


def rfc_model_builder(**model_params: Dict[str, Union[str, int, float]]) -> dc.models.SklearnModel:
    """
    Function that initializes a random forest classifier model given model parameters
    :param model_params: A dictionary that contains model parameters for a random forest classifier model
    :return: A deepchem scikit-learn model object
    """

    # maximum number of features to sample from 
    max_features = model_params['max_features']
    n_estimators = model_params['n_estimators']

    # constructing model with specified hyperparameter
    rfc = RandomForestClassifier(max_features = max_features,n_estimators=n_estimators)

    #Wrapping scikit-learn estimator in deepchem model class
    deepchem_rfc_model = dc.models.SklearnModel(rfc)

    return deepchem_rfc_model


def xgb_model_builder(**model_params: Dict[str, Union[str, int, float]]) -> dc.models.SklearnModel:
    """
    # Function that initializes a xgboost model given model parameters
    :param model_params: A dictionary that contains hyperparameters values 
    :return: A deepchem scikit-learn model object
    """

    #shrinkage rate, stage-wise scaling of the contribution of each tree 
    eta = model_params['eta']

    #L2 penalty on leaf values of trees
    lambda_ = model_params['reg_lambda']

    #subsample ratio
    subsample = model_params['subsample']

    #initializing xgboost classifier 
    xgb_model = xgb.XGBClassifier(eta = eta, reg_lambda = lambda_, subsample = subsample)

    #wrapping xgboost model in deepchem model class
    deepchem_xgb_model = dc.models.SklearnModel(xgb_model)

    return deepchem_xgb_model

class NN_Model(Model):
    """Template for constructing Keras models with three hidden layers with specified dropout applied to the activations of the first two layers, 
        and a specified L2 penalty on weights
    """

    def __init__(self, dropout: float,weights_penalty: float):
        """
        :param dropout: A float that represents the dropout rate for nodes during training
        :param weights_penalty: A float that represents the coefficient of the L2 norm on weights added to the objective function
        """

        super(NN_Model, self).__init__()
        self.layer_1 = layers.Dense(1000, activation='relu', kernel_regularizer = regularizers.L2(weights_penalty))
        self.layer_2 = layers.Dense(250, activation='relu', kernel_regularizer = regularizers.L2(weights_penalty))
        self.layer_3 = layers.Dense(50, activation='relu', kernel_regularizer = regularizers.L2(weights_penalty))
        self.layer_4 = layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.L2(weights_penalty))
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.layer_1(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        #x = self.dropout(x)
        x = self.layer_4(x)
        return x
    
def nn_model_builder(**model_params: Dict[str, Union[str, int, float]]) -> dc.models.KerasModel:
    """
    Function that initializes a neural network model given hyperparameters
    :param model_params: A dictionary that contains hyperparameters values
    :return: A deepchem keras model object
    """

    dropout  = model_params['dropout']
    weights_penalty = model_params['weights_penalty']

    #Specifying optimizer and learning rate
    adam_optimizer = dc.models.optimizers.Adam(0.001)

    #Initializing self-defined Keras model
    nn_model_instance = NN_Model(dropout,weights_penalty)

    #wrapping keras model in deepchem model class
    deep_chem_keras_model = dc.models.KerasModel(nn_model_instance,loss = dc.models.losses.BinaryCrossEntropy(), optimizer= adam_optimizer)

    return deep_chem_keras_model


def cartesian_dictionary(params_dict: Dict) -> Generator[Dict[str, Union[str, int, float]],None,None]:
    '''
    Generator that produces all possible combinations of hyperparameters 
    :param params_dict: Dictionary thats maps hyperparameter names to a list of possible values
    :return: A generator that yields parameter dictionaries
    '''
    keys, values = params_dict.keys(), params_dict.values()
    products = itertools.product(*values)

    for c in products:
        yield dict(zip(keys, c))

def hyperparameter_gridsearch_with_CV(model_builder: Callable,
                                      k_fold_datasets: List[dc_dataset],
                                      params_dictionary: Dict[str, Union[str, int, float]],
                                      metric: dc.metrics.Metric) -> Tuple[Dict[str, Union[str, int, float]], Dict[str, float],List[dc_model]]:
    """
    Function that searches for optimal hyperparameters with k-fold cross validation strategy
    :param model_builder: A function that initializes and returns a deepchem model
    :param k_fold_datasets: List of deepchem datasets
    :param params_dictionary: Dictionary thats maps hyperparameter names to a list of possible values
    :param metric: A metric used to evaluate model performance for a set of hyperparameters (examples: ROC_AUC score, accuracy)
    :return: A triple (Dictionary of optimal hyperparameters, Dictionary of all results, Models fitted on respective k-1 folds)
    """

    #initialize parameter generator
    generator = cartesian_dictionary(params_dictionary)

    #optimal hyperparameters
    optimal_hyperparameters = None

    #initialize dictionary, where the keys are a set of hyperparameters and the values are the averaged validation scores across the folds
    metric_results = {}

    #list of models fitted on respective k-1 folds (i.e training sets) with optimal hyperparameters
    k_fold_fitted_models = None

    #best averaged cross validation score
    best_avg_cv_score = -math.inf

    #iterate through hyperparameter combinations
    for hyperparameters in generator:

        #initialize metric values list
        metric_values = []

        #List that collects trained models
        k_fold_fitted_models_tmp = []
        
        #iterate through data_sets:
        for data_set in k_fold_datasets:
            train, valid = data_set

            #initialize\build model with given hyperparameters
            model = model_builder(**hyperparameters)

            #fit model on train dataset
            model.fit(train)

            #store model 
            k_fold_fitted_models_tmp.append(model)

            #evaluate fitted model on validation dataset and store accuracy
            metric_tmp = model.evaluate(valid,metric)

            #model.evaluate returns a dictionary of the form {metric_name: metric_value}
            metric_value = list(metric_tmp.values())[0]
            metric_values.append(metric_value)
            
        # computing averaged score
        n = len(metric_values)
        current_avg_cv_score = (1/n)*np.sum(metric_values)

        # converting hyperparameter dictionary to string in order to store as key of results dictionary
        hyperparameters_string = str(hyperparameters)
         # storing current score in results dictionary
        metric_results[hyperparameters_string] = current_avg_cv_score

        #updates best score, fitted models list, and optimal hyperparameters if current score is larger then best score
        print(best_avg_cv_score,current_avg_cv_score,hyperparameters)
        if best_avg_cv_score < current_avg_cv_score:
            best_avg_cv_score = current_avg_cv_score
            k_fold_fitted_models = k_fold_fitted_models_tmp.copy()
            optimal_hyperparameters = hyperparameters

    return (optimal_hyperparameters,metric_results,k_fold_fitted_models)


#Plotting Functions
def get_fold_scores(models_list: List[dc_model], 
                    k_folds_datasets: List[dc_dataset], 
                    metric: dc.metrics.Metric) -> List[Tuple[float, float]]:
    
    """
    Function that retrieves metric scores for training sets and validation sets
    :models_list: list of fitted models on respective data_set
    :param k_folds_datasets: List of deep chem datasets
    :param metric: A metric to evaluate model performance
    """

    scores = []
    num_folds = len(k_folds_datasets)

    for index in range(num_folds):

        train, valid = k_folds_datasets[index]

        model = models_list[index]

        score_tmp_train = model.evaluate(train,metric)
        score_train = list(score_tmp_train.values())[0]
        score_tmp_valid = model.evaluate(valid,metric)
        score_valid = list(score_tmp_valid.values())[0]
        
        scores.append((score_train,score_valid))

    return scores

def generate_roc_plot(fitted_model_dict: Dict[str,dc_model], eval_data: dc_dataset, save_path: str = None):
    """
    Function that produces a plot of ROC curves corresponding to each model, along with the AUC values

    :param fitted_model_dict: A dictionary that maps model name to fitted model
    :param eval_data: A deepchem dataset that is used to evaluate fitted models 
    :param save_path: String that represents path in which the plot will be saved
    """

    sns.set_style("darkgrid")
    sns.set_palette("viridis")

    for model_type in fitted_model_dict.keys():
        y_prob = None
        y_true = None

        model = fitted_model_dict[model_type]

        # This distinction is needed because NN only outputs the positive class probability instead of both classes
        if model_type == "Neural Network":
            y_prob = model.predict(eval_data)
            y_true = eval_data.y

        else:
            #Extracting positive class probabilities
            y_prob = model.predict(eval_data)[:,1]
            y_true = eval_data.y 

        #using scikit-learn metric for ROC_Curve data
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_prob)

        # ROC_AUC, Accuracy metrics
        roc_auc_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        accuracy_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

        scores = model.evaluate(eval_data, [roc_auc_metric,accuracy_metric])
        roc_auc = scores['roc_auc_score']
        accuracy = scores['accuracy_score']

        plt.plot(fpr,tpr,label = model_type + ", ROC AUC Score="+ str(round(roc_auc,3)) + ", Accuracy =" + str(round(accuracy,3)))


    #Adds diagonal line for base-line
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1]) 

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Test Set')

    plt.legend()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def generate_metric_barplot(fitted_models_dict: Dict[str, List[dc_model]], 
                            k_folds_datasets: List[dc_dataset],
                            metric_name: str,
                            save_path: str = None, ):
    """
    Function that produces a bar plot containing train and validation ROC-AUC scores 
    on respective datasets in k-fold of each model supplied in the model list
    :param fitted_models_dict: Dictionary that maps model name to a list of fitted models
    :param k_folds_datasets: List of deep chem datasets
    :param save_path: String that represents path in which plot will be saved
    """

    sns.set_style("darkgrid")
    sns.set_palette("viridis")


    TRAIN_IDX = 0
    VAL_IDX = 1

    metric = None
    if metric_name == "ROC_AUC":
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    if metric_name == "Accuracy":
        metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    metric_values = []
    
    for model_type, models_list in fitted_models_dict.items():

        # return list of tuples of scores corresponding to the train set and validation set for each partition of the k-fold
        fold_scores = get_fold_scores(models_list, k_folds_datasets, metric)

        num_folds = len(k_folds_datasets)

        #fold_num is the fold number that is treated as the validation set
        for fold_num in range(num_folds):
            for split in [TRAIN_IDX, VAL_IDX]:

                split_label = ""

                if split == TRAIN_IDX:
                    split_label = "Train"
                else:
                    split_label = "Validation"

                metric_values.append((model_type,
                                      split_label, 
                                      "Fold %d" % fold_num, 
                                      fold_scores[fold_num][split]))
                

    df = pd.DataFrame(metric_values)
    df.columns = ["Model_Type", "Split", "Fold_Num", "Fold_Scores"]

    sns.barplot(data=df, x="Fold_Scores", y="Split", errorbar=("sd"), 
                capsize=.1, errcolor=".05", errwidth = 1.5, edgecolor=".1", hue = "Model_Type")
    
    
    plt.xlabel(metric_name + ' Score')
    plt.ylabel('')
    plt.title('Average Train and Validation Set Scores Across Folds')

    if save_path:
        plt.savefig(save_path) 

    plt.close()


# File and Pickling Functions 
def make_dir(path: str):
    """
    Function that creates a folder given the specified path
    :param path: A str that designates the path of the folder to be created
    """

    if not os.path.exists(path):
        os.mkdir(path)
        print("Folder %s created" % path)
    else:
        print("Folder %s already exists" % path)  

def save_hyper_params(file_path: str, hyper_params):
    """
    Function that saves a hyperparameters to a specified file path
    :param file_path: A str file_path where the dictionary will be saved
    :param hyper_params: A dictionary of model hyperparameters
    """

    with open(file_path, 'wb') as f:
        pickle.dump(hyper_params, f)


def load_hyper_params(file_path: str):
    """
    Function that loads a hyperparameters from a specified file path
    
    :param file_path: A str file_path where the dictionary will be loaded from
    :return: A dictionary of model hyperparameters
    """

    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def main():

    print("Making folders")
    make_dir("./saved_hyperparameters")
    
    print("Loading in Dataset")
    bbbp_data = pd.read_csv('data/BBBP.csv')

    print("Loading splitters and featurizers")
    circular_fingerprint = dc.feat.CircularFingerprint(size=2048, radius=2)
    scaffold_splitter = dc.splits.ScaffoldSplitter()

    print("Featurizing dataset with circular fingerprint")
    data_representation = construct_representation(bbbp_data,circular_fingerprint)
    
    #Splits data into test and train
    train, test =  scaffold_splitter.train_test_split(data_representation)

    # Splits train data in k-folds
    k_fold_datasets = k_fold_data_splitting(dataset = data_representation,splitter = scaffold_splitter,folds = 4)

    #Selecting accuracy metric to train and tune models by
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    #Logistic Regression
    print("Searching for optimal hyperparameters for Logisitc Regression model")
    logistic_optimal_hyperparameters, logistic_all_results,logistic_fitted_kfold_models = hyperparameter_gridsearch_with_CV(logistic_model_builder,k_fold_datasets,LOGISTIC_PARAMS,metric)
    save_hyper_params("./saved_hyperparameters/logistic_hyperparams_optimal.pkl", logistic_optimal_hyperparameters)
    save_hyper_params("./saved_hyperparameters/logistic_hyperparams_all.pkl", logistic_all_results)

    print("Fitting logistic regression model using optimal hyperparameters")
    logistic_model = logistic_model_builder(**logistic_optimal_hyperparameters)
    logistic_model.fit(train)


    #Random Forest
    print("Searching for optimal hyperparameters for Random Forest Classifier model")
    rfc_optimal_hyperparameters, rfc_all_results, rfc_fitted_kfold_models = hyperparameter_gridsearch_with_CV(rfc_model_builder,k_fold_datasets,RFC_PARAMS,metric)
    save_hyper_params("./saved_hyperparameters/rfc_hyperparams_optimal.pkl", rfc_optimal_hyperparameters)
    save_hyper_params("./saved_hyperparameters/rfc_hyperparams_all.pkl", rfc_all_results)

    print("Fitting Random Forest model using optimal hyperparameters")
    rfc_model = rfc_model_builder(**rfc_optimal_hyperparameters)
    rfc_model.fit(train)


    #Xgboost
    print("Searching for optimal hyperparameters for XGBoost Classifier model")
    xgb_optimal_hyperparameters, xgb_all_results, xgb_fitted_kfold_models = hyperparameter_gridsearch_with_CV(xgb_model_builder,k_fold_datasets,XGB_PARAMS,metric)
    save_hyper_params("./saved_hyperparameters/xgb_hyperparams_optimal.pkl", xgb_optimal_hyperparameters)
    save_hyper_params("./saved_hyperparameters/xgb_hyperparams_all.pkl", xgb_all_results)

    print("Fitting XGBoost model using optimal hyperparameters")
    xgb_model = xgb_model_builder(**xgb_optimal_hyperparameters)
    xgb_model.fit(train)


    #NN 
    print("Searching for optimal hyperparameters for NN model")
    nn_optimal_hyperparameters, nn_all_results, nn_fitted_kfold_models = hyperparameter_gridsearch_with_CV(nn_model_builder,k_fold_datasets,NN_PARAMS,metric)
    save_hyper_params("./saved_hyperparameters/nn_hyperparams_optimal.pkl", nn_optimal_hyperparameters)
    save_hyper_params("./saved_hyperparameters/nn_hyperparams_all.pkl", nn_all_results)

    print("Fitting NN model using optimal hyperparameters")
    nn_model = nn_model_builder(**nn_optimal_hyperparameters)
    nn_model.fit(train)


    #Models which have been fitted on all training data using optimal hyperparameters
    optimal_model_dict = {
        "Logistic Regression": logistic_model,
        "Random Forest": rfc_model,
        "XGBoost": xgb_model,
        "Neural Network": nn_model
    }

    #Models which have been fitted on respective k-1 folds during grid search with k-fold cross validation
    fitted_models_dict = {
        "Logistic Regression": logistic_fitted_kfold_models,
        "Random Forest": rfc_fitted_kfold_models,
        "XGBoost": xgb_fitted_kfold_models,
        "Neural Network": nn_fitted_kfold_models
    }

    print("Creating and saving ROC_AUC Plots")
    generate_roc_plot(optimal_model_dict, test, save_path="./roc_plot.png")
    generate_metric_barplot(fitted_models_dict, k_fold_datasets, 'ROC_AUC', save_path="./avg_test_validation.png")

if __name__ == '__main__':
    main()
