import pandas as pd
PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/wszystkie_csv.csv'

names = [
    [
        'CH4_NKZG_AN1_METAN',
        'CH4_NKZG_AN1_AZOT',
        'CH4_NKZG_AN1_WART_OPAL',
        'S0FK318',
        'CH4_NKZG_AN2_METAN',
        'CH4_NKZG_AN2_AZOT',
        'CH4_NKZG_AN2_WART_OPAL',
        'S0FC027.PIDA.PV',
        'S0PC025.PIDA.PV',
        'S0F026',
        'S0T001_1',
        'S0FK319',
        'S0PC318.PIDA.PV',
        'S0FC301.PIDA.PV',
        'S0FC301.PIDA.SP',
        'S0FC301.PIDA.OP',
        'S0PC304.PIDA.PV',
        'S0RFC322.PIDA.PV',
        'S0RFC316.CTLALGO.PV',
        'S0FC316.PIDA.PV',
        'S0PC304.PIDA.SP',
        'S0PC304.PIDA.OP',
        'S0PC027.PIDA.PV',
        'S0T302_2',
        'S0T302_2A',
        'S0A301',
        'S0T302_S14',
        'S0T302_S15',
        'S0T302_S16',
        'S0T302_S17',
        'S0T302_S18',
        'S0T302_S19',
        'S0P304_2',
        'S0T301_6',
        'S0P302',
        'S0PZ304.DACA.PV'
    ]
]


def main():
    df = pd.read_csv(PATH,'Arkusz1', parse_col=names[0])
    print(df)

if __name__ == '__main__':
    main()