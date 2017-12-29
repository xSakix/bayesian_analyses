import pandas
import os
import quandl


def load_data(assets):
    file = 'crypto_data.csv'

    if not os.path.isfile(file):
        loaded_data = quandl.get(assets)
        loaded_data.to_csv(file)

    return pandas.read_csv(file).Last
