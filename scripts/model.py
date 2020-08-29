# models + accuracy metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve, GridSearchCV, StratifiedKFold

# evaluation
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

import matplotlib.pyplot as plt


# train and test sets
def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state = 42)
    return x_train, x_test, y_train, y_test

# cross validation

# kfold
kf = KFold(n_splits=10, random_state = 42)
def overall_score(model, x_train, y_train):
    accuracy = cross_val_score(estimator = model, X = x_train, y = y_train, cv = kf)
    return accuracy.mean()

#stratified
cv = StratifiedKFold(n_splits=10, random_state=42)
def overall__stratified_score(model, x_train, y_train):
    accuracy = cross_val_score(estimator = model, X = x_train, y = y_train, cv = cv)
    return accuracy.mean()

# models
class Model(x, y):
    x_train, x_test, y_train, y_test = split(x, y)

    # logistic regression
    def logit(self):
        regressor = LogisticRegression(C = 1000, penalty= 'l2')
        regressor = regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        print("Logistic Regression Accuracy score:", accuracy_score(pred, y_test))
        return pred

    # multilayer percpeptron
    def mlp(self):
        mlp = MLPClassifier(C = 1000, penalty= 'l2')
        mlp = mlp.fit(x_train, y_train)
        pred = mlp.predict(x_test)
        print("MLP's Accuracy score:", accuracy_score(pred, y_test))
        return pred

    #xgboost
    def xgb(self):
        xgb = XGBClassifier(silent = True,max_depth = 6, n_estimators = 200)
        xgb = xgb.fit(x_train, y_train)
        pred = xgb.predict(x_test)
        print("XGBoost's Accuracy score:", accuracy_score(pred, y_test))
        return pred

class Evaluation():

    def precision_recall_f1_support(self, y_test, pred):
        precision, recall, fscore, support = score(y_test, pred)
        return precision, recall, fscore, support

    def roc_plot(self, model, x_test, y_test):
        #predict probabilities
        ns_probs = [0 for _ in range(len(x_test))]  # no skill
        m_prob = model.predict_proba(x_test)

        # keep probabilities for the positive outcome only
        m_prob = m_prob[:, 1]

        # calculate scores then print them
        m_auc = roc_auc_score(y_test, m_prob)
        print('model:', m_auc)

        # calculate roc curves
        m_fpr, m_tpr, _ = roc_curve(y_test, m_prob)
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

        # plot the roc curve for the model
        plt.figure(figsize = (12,7))
        plt.plot(m_fpr, m_tpr, marker='.', label='model')
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

        plt.legend()
        plt.title('ROC curves for the models')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()