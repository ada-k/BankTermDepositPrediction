#!/usr/bin/env python
from data import DataPrep, Transform
from model import Model, Evaluation, split, overall_score, overall__stratified_score

import sys

# data loading + preprocessing
def main():
    # instantiate the class
    d = DataPrep()

    # read the data
    # path = '/content/bank-additional-full.csv'
    path = sys.argv[1]
    data = d.read_data(path)
    print('Original shape:', data.shape)

    # preprocessing 
    data = d.treat_null(data)
    data = d.outlier_correcter(data)
    data = d.generate_features(data)
    print('After feature generation:', data.shape)
    data = d.scaler(data)
    print('After scaling:', data.shape)
    data = d.encoder(data)
    print('After encoding:', data.shape)
    data = d.over_sample(data)
    print('After resampling:', data.shape)
    data = drop_unwanted(data)
    print('After dropping unwanted features:', data.shape)
    print(data.head())

    # split data
    t = Transform()
    x, y = t.split(data)


    # modeling
    m = Model(x, y)
    # using mlp (best predictor of the 3)
    pred = m.mlp()
    pred_df = pd.DataFrame(pred, columns = ['y']) # save the predictions to a df
    pred_df.to_csv('pred.csv') # save predictions to csv


    # evaluation
    x_train, x_test, y_train, y_test = split(x, y)
    e = Evaluation()
    precision, recall, fscore, support = e.precision_recall_f1_support(y_test, pred)
    print('precision:', precision)
    print('precision:', precision)
    print('precision:', precision)
    print('precision:', precision)

if __name__=="__main__": 
    main() 
