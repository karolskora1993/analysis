import pandas as pd
import numpy as np
from analysis import blocks


def main():
    df = pd.read_excel(blocks.PATH, 'Arkusz1')
    print("dataframe created")
    series = [df[s] for s in blocks.names[0]]
    print(series)
    resultDf = pd.DataFrame(np.asarray(series), columns=names[0])
    print(resultDf)

if __name__ == '__main__':
    main()