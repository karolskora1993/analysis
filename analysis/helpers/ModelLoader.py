from keras.models import load_model
from time import time
import os

def load(path):
    start = time()
    model = load_model(path)
    print('time:{0}'.format(time()-start))
    return model


if __name__ == '__main__':
    home = str(os.path.expanduser('~') + '/')
    path = home + 'Dokumenty/analysis/data/models/blok III_S0T302_5.DACA.PV_KerasConv_().h5'
    load(path)

