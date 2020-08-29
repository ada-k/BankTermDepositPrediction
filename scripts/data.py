# Linear algebra + data processing
import pandas as pd
import numpy as np

# preprocessing 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

# read and process data
class DataPrep():

    # read the data
    def read_data(self, path):
        data = pd.read_csv(path, sep = ';')
        return data


    def treat_null(self, data):
        global categorical, discrete, continous, cols
        categorical = []
        discrete = []
        continous = []
        for col in data.columns:
            if data[col].dtype == object:
                categorical.append(col)
            elif data[col].dtype in ['int16', 'int32', 'int64']:
                discrete.append(col)
            elif data[col].dtype in ['float16', 'float32', 'float64']:
                continous.append(col)

        cols = discrete + categorical + continous
        data = data[cols]

        # null values
        # data = preprocess_data(data)
        indices = []
        for col in cols:
            k = data.columns.get_loc(col)
            indices.append(k)

        for col in indices:
            if data.columns[col] in discrete:
                x = data.iloc[:, col].values
                x = x.reshape(-1,1)
                imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                imputer = imputer.fit(x)
                x = imputer.transform(x)
                data.iloc[:, col] = x

            if data.columns[col] in continous:
                x = data.iloc[:, col].values
                x = x.reshape(-1,1)
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer = imputer.fit(x)
                x = imputer.transform(x)
                data.iloc[:, col] = x

            elif data.columns[col] in categorical:
                x = data.iloc[:, col].values
                x = x.reshape(-1,1)
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imputer = imputer.fit(x)
                x = imputer.transform(x)
                data.iloc[:, col] = x     

        return data

    # outlier detection + treatment
    def outlier_correcter(self, data):
        # data = treat_null(data)
        for col in discrete + continous:
            data[col] = data[col].clip(lower=data[col].quantile(0.10), upper=data[col].quantile(0.90))
        return data

    # feature generation
    def generate_features(self, data):
        data['both_loans'] = 0 #default to 0
        data.loc[data['housing'] == 'yes', 'both_loans'] = 1
        data.loc[data['loan'] == 'no', 'both_loans'] = 1 # change to 1 if one has both loans
        data['total_contacts'] = data['campaign'] + data['previous']

        def squares(data, ls):
            m = data.shape[1]
            for l in ls:
                # data = data.assign(newcol=pd.Series(np.log(1.01+data[l])).values)
                data = data.assign(newcol=pd.Series(data[l]*data[l]).values)    
                data.columns.values[m] = l + '_sq'
                m += 1
            return data

        log_features = ['duration', 'cons.price.idx', 'emp.var.rate', 'cons.conf.idx', 'euribor3m']

        data = squares(data, log_features)

        return data

    # scaling numerical
    def scaler(self, data):
        # data = outlier_correcter(data)
        indices = []
        for col in discrete + continous + ['total_contacts', 'duration_sq', 'cons.price.idx_sq', 'emp.var.rate_sq', 'cons.conf.idx_sq', 'euribor3m_sq']:
            k = data.columns.get_loc(col)
            indices.append(k)

        for col in indices:
            x = data.iloc[:, col].values
            x = x.reshape(-1,1)
            imputer = StandardScaler()
            imputer = imputer.fit(x)
            x = imputer.transform(x)
            data.iloc[:, col] = x  

        return data

    # encoding categorical
    def encoder(self, data):
        # data = scaler(data)
        cols = categorical.copy()
        cols.remove('y')
        data = pd.get_dummies(data, columns = cols)
        return data

    # class imbalance
    def over_sample(self, data):
        # data = scaler(data)
        subscribers = data[data.y == 'yes']
        non_subscribers = data[data.y == 'no']

        subscribers_upsampled = resample(subscribers, replace=True, # sample with replacement
                            n_samples = len(non_subscribers), # match number in majority class
                            random_state = 42) # reproducible results

        data = pd.concat([subscribers_upsampled, non_subscribers])
        return data

    def drop_unwanted(data):
        data = data.drop(['duration', 'duration_sq'], axis = 1)
        return data



# feature transformations

# target and predictor variables
class Transform():
    def split(self, data):
        x = data.drop(['y'], axis = 1)
        y = data[['y']]
        # Encode labels in target df. 
        encoder = LabelEncoder() 
        y = encoder.fit_transform(y) 
        y = pd.DataFrame(y)
        y.columns = ['y']
        return x, y

    # x, y = split(data)

    # pca
    def pca_transform(self, x):
        pca = PCA(n_components = 8)
        pca.fit(x)
        pca_scores = pca.transform(x)
        x_pca = pd.DataFrame(pca_scores)
        return x_pca

    # tsne
    # tsne with pca
    def tsne_with_pca(self, pca_scores):
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_scores)
        x_tsne_pca = pd.DataFrame(columns = ['tsne_pca_one', 'tsne_pca_two'])
        x_tsne_pca['tsne_pca_one'] = tsne_pca_results[:,0]
        x_tsne_pca['tsne_pca_two'] = tsne_pca_results[:,1]
        return x_tsne_pca

    # tsne on original dataset
    def tsne(self, x):
        tsne = TSNE(n_components = 2)  #using 2 components since the barnes_hut algorithm relies on quad-tree or oct-tree.
        tsne_results = tsne.fit_transform(x)
        x_tsne_org = pd.DataFrame(columns = ['tsne_one', 'tsne_two'])
        x_tsne_org['tsne_one'] = tsne_results[:,0]
        x_tsne_org['tsne_two'] = tsne_results[:,1]
        return x_tsne_org
