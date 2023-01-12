import pandas as pd 
import numpy as np

# считываем данные
data = pd.read_csv('./creditcard — copy.csv', index_col=[0])

# извлекаем список колонок
data_cols = list(data.columns)
print('Dataset columns: {}'.format(data_cols))

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def transformations(data):
    processed_data = data.copy()
    data_cols = list(data.columns[data.columns != 'Class'])

    data_transformer = Pipeline(steps=[
        ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True, copy=True))])

    preprocessor = ColumnTransformer(
        transformers = [('power', data_transformer, data_cols)])
    processed_data[data_cols] = preprocessor.fit_transform(data[data_cols])

    return data, processed_data, preprocessor
    
_, data, preprocessor = transformations(data)

# импорт функций преобразования
!pip install ydata-synthetic
from ydata_synthetic.preprocessing.regular import *

# мы синтезируем только класс мошеннических транзакций
# в тренировочных данных содержится всего 492 таких объекта (из 285 тыс.)
train_data = data.loc[ data['Class']==1 ].copy()

# перед тренировкой GAN(генеративно-состязательной сети) преобразовываем данные
# применим преобразование, чтобы распределить данные используя нормальное распределение
data = transformations(train_data)

print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))

from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# определяем GAN и тренировочные параметры
noise_dim = 32
dim = 128
batch_size = 128

log_step = 20
epochs = 60+1
learning_rate = 5e-4
beta_1 = 0.5
beta_2 = 0.9

gan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, betas=(beta_1, beta_2),
                           noise_dim=noise_dim, n_cols=train_data.shape[1], layers_dim=dim)
train_args = TrainParameters('', epochs=epochs, sample_interval=log_step)
num_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 
            'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
            'V26', 'V27', 'V28', 'Amount']
cat_cols = ['Class']
# тренируем модель GAN, синтезируем данные
model = WGAN_GP
synthesizer = model(gan_args, n_critic=5)
synthesizer.train(train_data, train_args, num_cols, cat_cols)
