import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from helpers.DataHandler import save_model
import os

HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_v4/'
SCORE_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe/'
MODEL_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/'

BLOCKS = {
    # 'blok I': 23,
    'blok II': 36,
    'blok III': 21,
    'blok IV': 28
}



def calculate_Rkw(y, y_hat, y_mean):
    tot = y-y_mean
    tot = tot*tot
    ss_tot = tot.sum()
    res = y-y_hat
    res = res*res
    ss_res = res.sum()
    Rkw = 1 - ss_res/ss_tot
    return Rkw

def load_data(block_name, load_path, InVars = 23):
    df = pd.read_csv(load_path + block_name + '.csv', index_col=0)
    TrOut = df.iloc[:,InVars+1:-1]
    TrIn = df.iloc[:,1:InVars + 1]
    return TrIn, TrOut

def findBestRegModel(TrIn1, y, f, delay = 1):
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

    reg = reg.fit(pd.DataFrame(Tr2.iloc[:205039 - delay, ]), pd.DataFrame(y.iloc[:205039 - delay]))
    y_hat = reg.predict(Tr2.iloc[257134 - delay:, ])
    y_hat.shape = (-1,)
    Rkw = calculate_Rkw(y.iloc[257134 - delay:], y_hat, y.iloc[:205039 - delay].mean())
    f.write(str(Tr2.columns.tolist()) + '\n')
    f.write(str(Rkw) + '\n\n')
    print(str(Tr2.columns.tolist()) + '\n')
    print(str(Rkw) + '\n\n')

    return reg

def normalize(X, test_end):
    for i in range(X.shape[1]):
        mean =np.mean(X.iloc[:test_end, i])
        stdev = np.std(X.iloc[:test_end, i])
        X.iloc[:, i] = (X.iloc[:, i] - mean) / stdev
    return X

def delay(TrIn, TrOut):
    TrIn1 = TrIn.copy()
    TrIn1 = TrIn1.iloc[1:]
    TrIn1.reset_index(inplace=True)
    TrIn.columns = TrIn.columns + 'd1'
    TrIn = pd.concat([TrIn1, TrIn.iloc[:-1]], axis = 1)
    TrOut = TrOut.iloc[1:]
    TrOut.reset_index(inplace=True)
    return TrIn.iloc[:,1:], TrOut.iloc[:,1:], 1

def transfo(X):
    temp = X.copy()
    diffs = [1, 2, 3, 4, 5]
    rolls = [5, 10, 15]

    for df in diffs:
        for colname in temp.columns:
            z = temp[colname].copy()
            z.name = z.name + '_df' + str(df)
            z = z.diff(df)
            X = pd.concat([X, z], axis=1)

    for roll in rolls:
        for colname in temp.columns:
            mean_col = temp[colname].rolling(roll, 1).mean()
            mean_col.name = colname + '_r' + str(roll) + '_mean'
            X = pd.concat([X, mean_col], axis=1)

            min_col = temp[colname].rolling(roll, 1).min()
            min_col.name = colname + '_r' + str(roll) + '_min'
            X = pd.concat([X, min_col], axis=1)

            max_col = temp[colname].rolling(roll, 1).max()
            max_col.name = colname + '_r' + str(roll) + '_max'
            X = pd.concat([X, max_col], axis=1)

    X = X.iloc[15:]
    X.reset_index(inplace=True)
    return X


def main():

    for block, vars_in in BLOCKS.items():
        TrIn, TrOut = load_data(block, LOAD_PATH, vars_in)
        delay = 0
        TrIn = normalize(TrIn, test_end=205039)

        # TrIn, TrOut, delay= delay(TrIn, TrOut)
        TrIn, TrOut, delay= transfo(TrIn, TrOut)


        filename = SCORE_SAVE_PATH + 'forward_selection_{blok}.txt'.format(blok=block)
        f = open(filename, "a")
        for i in range(TrOut.shape[1]):
            print(TrOut.columns[i])
            f.write(TrOut.columns[i] + ':\n\n')
            model = findBestRegModel(TrIn, TrOut.iloc[:,i], f, delay=delay)
            save_path = MODEL_SAVE_PATH + '{0}_{1}_forward_reg.p'.format(block, TrOut.columns[i])
            save_model(model, save_path)

        f.close()


main()