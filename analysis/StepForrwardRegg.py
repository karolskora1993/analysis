import numpy as np
import pandas as pd
from sklearn import linear_model
from helpers.DataHandler import save_model, load_block_vars
import os
from sklearn.decomposition import PCA


HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe_bloki/serialized/'
SCORE_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe_bloki/'
BLOCK_VARS_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_poprawione_v5_temp.xlsx'


BLOCKS = [
    # 'blok I',
    # 'blok II',
    # 'blok III',
    'blok IV'
]

COL_NAMES = ['in', 'control', 'out', 'delay']



def calculate_Rkw(y, y_hat, y_mean):
    tot = y-y_mean
    tot = tot*tot
    ss_tot = tot.sum()
    res = y-y_hat
    res = res*res
    ss_res = res.sum()
    Rkw = 1 - ss_res/ss_tot
    return Rkw

def load_data(block_name, load_path):
    df = pd.read_csv(load_path + block_name + '.csv', index_col=0)
    var_names = load_block_vars(BLOCK_VARS_PATH)[block_name][COL_NAMES]

    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    delays = var_names['delay']
    return df[vars_in], df[vars_out], delays

def findBestRegModel(TrIn1, y, f, delay = 0, use_pca = False):
    TrIn = TrIn1.copy()
    bestRkw = -100
    bestRRkw = -100
    best_i = -1
    for i in range(TrIn.shape[1]):
        reg = linear_model.LinearRegression()
        In = pd.DataFrame(TrIn.iloc[:205039 - delay, i])
        reg = reg.fit(In, pd.DataFrame(y.iloc[:205039 - delay]))
        y_hat = reg.predict(pd.DataFrame(TrIn.iloc[:205039 - delay, i]))
        y_hat.shape = (-1,)
        RRkw = calculate_Rkw(y.iloc[:205039 - delay], y_hat, y.iloc[:205039 - delay].mean())
        y_hat = reg.predict(pd.DataFrame(TrIn.iloc[205039 - delay:257134 - delay, i]))
        y_hat.shape = (-1,)
        Rkw = calculate_Rkw(y.iloc[205039 - delay:257134 - delay], y_hat, y.iloc[:205039 - delay].mean())
        if Rkw>bestRkw:
            bestRRkw = RRkw
            bestRkw = Rkw
            best_i = i
    Tr2 = TrIn.iloc[:,best_i]
    improvement = True
    count = 0
    while improvement and count<15:
        improvement = False
        TrIn.drop(TrIn.columns[best_i], 1, inplace=True)
        for i in range(TrIn.shape[1]):
            reg = linear_model.LinearRegression()
            TrIn2 = pd.concat([Tr2, TrIn.iloc[:,i]], axis = 1)
            reg = reg.fit(TrIn2.iloc[:205039 - delay,], pd.DataFrame(y.iloc[:205039 - delay]))
            y_hat = reg.predict(TrIn2.iloc[:205039 - delay,])
            y_hat.shape = (-1,)
            RRkw = calculate_Rkw(y.iloc[:205039 - delay], y_hat, y.iloc[:205039 - delay].mean())
            y_hat = reg.predict(TrIn2.iloc[205039 - delay:257134 - delay,])
            y_hat.shape = (-1,)
            Rkw = calculate_Rkw(y.iloc[205039 - delay:257134 - delay], y_hat, y.iloc[:205039 - delay].mean())
            if Rkw > bestRkw:
                bestRRkw = RRkw
                bestRkw = Rkw
                best_i = i
                improvement = True
        if improvement:
            Tr2 = pd.concat([Tr2, TrIn.iloc[:, best_i]], axis=1)
            count += 1

    x = Tr2.iloc[:205039 - delay, ]
    pca = PCA(n_components=Tr2.shape[1] // 2)
    if use_pca:
        pca = pca.fit(Tr2.iloc[:205039 - delay, ])
        x = pca.transform(Tr2.iloc[:205039 - delay, ])

    reg = reg.fit(x, pd.DataFrame(y.iloc[:205039 - delay]))
    x_test = Tr2.iloc[257134 - delay:, ]
    if use_pca:
        x_test = pca.transform(x_test)
    y_hat = reg.predict(x_test)
    y_hat.shape = (-1,)
    Rkw = calculate_Rkw(y.iloc[257134 - delay:], y_hat, y.iloc[:205039 - delay].mean())
    f.write(str(Tr2.columns.tolist()) + '\n')
    f.write(str(Rkw) + '\n\n')
    f.write(str(pca.get_covariance) + '\n\n')
    print(str(Tr2.columns.tolist()) + '\n')
    print(str(Rkw) + '\n\n')

    return reg

def normalize(X, test_end):
    for i in range(X.shape[1]):
        mean =np.mean(X.iloc[:test_end, i])
        stdev = np.std(X.iloc[:test_end, i])
        X.iloc[:, i] = (X.iloc[:, i] - mean) / stdev
    return X


def transfo(TrIn, TrOut):
    X = TrIn.copy()
    diffs = [1, 2, 3, 4, 5]
    for df in diffs:
        for colname in X.columns:
            z = X[colname].copy()
            z.name = z.name + '_df' + str(df)
            z = z.diff(df)
            TrIn = pd.concat([TrIn, z], axis=1)

    rolls = [5, 10, 15]
    for roll in rolls:
        for colname in X.columns:
            Xr = X[colname].rolling(roll, 1).mean()
            Xr.name = colname + '_r' + str(roll) + '_mean'
            TrIn = pd.concat([TrIn, Xr], axis=1)

    rolls = [5, 10, 15]
    for roll in rolls:
        for colname in X.columns:
            Xr = X[colname].rolling(roll, 1).min()
            Xr.name = colname + '_r' + str(roll) + '_min'
            TrIn = pd.concat([TrIn, Xr], axis=1)

    rolls = [5, 10, 15]
    for roll in rolls:
        for colname in X.columns:
            Xr = X[colname].rolling(roll, 1).max()
            Xr.name = colname + '_r' + str(roll) + '_max'
            TrIn = pd.concat([TrIn, Xr], axis=1)

    TrIn = TrIn.iloc[15:]
    TrIn.reset_index(inplace=True)
    TrOut = TrOut.iloc[15:]
    TrOut.reset_index(inplace=True)
    return TrIn.iloc[:,1:], TrOut.iloc[:,1:]

def shift_data(input_data, output_data, delay):
    input_data = input_data[: -delay]
    output_data = output_data[delay:]
    return input_data, output_data

def main():

    for block in BLOCKS:
        TrIn, TrOut, delays = load_data(block, LOAD_PATH)

        TrIn = normalize(TrIn, test_end=205039)

        TrIn, TrOut = transfo(TrIn, TrOut)


        filename = SCORE_SAVE_PATH + 'forward_selection_{blok}_pca.txt'.format(blok=block)
        f = open(filename, "a")
        for i in range(TrOut.shape[1]):

            delay = int(delays[i]) if delays[i] >= 1 else 0

            if delay > 0:
                x, y = shift_data(TrIn, TrOut.iloc[:, i], delay)
            else:
                x, y = TrIn, TrOut.iloc[:, i]

            print(TrOut.columns[i])
            f.write(TrOut.columns[i] + ':\n\n')
            model = findBestRegModel(x, y, f, delay=delay, use_pca= True)
            save_path = MODEL_SAVE_PATH + '{0}_{1}_forward_reg_pca.p'.format(block, TrOut.columns[i])
            save_model(model, save_path)

        f.close()


main()