IN_DATA_LENGTH = {
    23: 'blokI',
    36: 'blokII',
    20: 'blokIII',
    28: 'blokIV'
}

PREDICTORS = {
    'S0T301_17.DACA.PV': ['S0PC020.PIDA.OP_r10_mean', 'S0A301.DACA.PV_r15_min', 'S0T302_S19.DACA.PV_r15_max', 'S0T302_2A.DACA.PV_r10_min', 'S0T301_6.DACA.PV', 'S0FC301.PIDA.PV_r5_min', 'S0A301.DACA.PV_r15_mean', 'S0T302_2A.DACA.PV_r10_mean', 'S0T302_S17.DACA.PV_r15_min', 'S0P304_2.DACA.PV_r15_max', 'S0FR004.DACA.PV_r15_max', 'S0TC304.PIDA.PV_r10_min', 'S0P314.DACA.PV_df5', 'S0T302_S17.DACA.PV_r5_min', 'S0T302_2A.DACA.PV_r15_mean', 'WOD_AZOT_1_df5'],
    'S0T302_1.DACA.PV': ['S0T302_2.DACA.PV', 'S0P304_2.DACA.PV_r15_min', 'S0PC020.PIDA.OP_r5_mean', 'S0T302_S18.DACA.PV', 'S0PC020.PIDA.OP_df4', 'S0T302_2A.DACA.PV_r15_mean', 'S0P311.DACA.PV_r15_max', 'S0T302_2A.DACA.PV_r10_min', 'S0T302_S16.DACA.PV_r15_min', 'S0TC304.PIDA.PV_r15_min', 'S0CH4_1.DACA.PV_r15_min', 'S0T302_S15.DACA.PV_df5', 'S0P314.DACA.PV_r10_min', 'S0FC301.PIDA.OP_r10_mean', 'S0T302_2A.DACA.PV_r15_max', 'S0T302_S16.DACA.PV_r15_max'],
    'S0T301_2.DACA.PV': ['S0TC304.PIDA.PV', 'S0T302_S18.DACA.PV', 'S0FR004.DACA.PV_r15_max', 'S0FC314.PIDA.PV_r5_min', 'S0T302_S14.DACA.PV_r15_max', 'S0T302_S15.DACA.PV_r10_min', 'S0A301.DACA.PV_r15_mean', 'S0T301_6.DACA.PV_r5_mean', 'S0FC301.PIDA.PV_r5_min', 'S0PC020.PIDA.OP_r15_max', 'S0P311.DACA.PV_r15_max', 'S0T306_1.DACA.PV_r15_max', 'S0PC020.PIDA.PV_r15_max', 'S0T302_S18.DACA.PV_r15_min', 'WOD_METAN_3_r15_min', 'S0T302_S16.DACA.PV_r15_min'],
    'S0CH4_6.DACA.PV':['S0PC306.PIDA.PV_r5_min', 'S0T301_15.DACA.PV_r15_mean', 'S0FC301.PIDA.PV_r15_mean', 'S0T301_15.DACA.PV_r10_mean', 'S0HV302_1.S0FV302_1.OP', 'S0T302_3.DACA.PV_r15_min', 'S0T301_15.DACA.PV_r5_mean', 'S0FC301.PIDA.SP', 'S0P312.DACA.PV_df1', 'S0T302_3.DACA.PV_df5', 'S0T301_18.DACA.PV_df4', 'S0HV302_1.S0FV302_1.OP_r10_max', 'S0T301_19.DACA.PV_df3', 'S0TC402.PIDA.PV_r10_max', 'S0FC301.PIDA.PV_r10_mean', 'S0FC301.PIDA.PV_r5_mean'],
    'S0N2_6.DACA.PV': ['S0CO_1.DACA.PV_r15_max', 'S0HV302_1.S0FV302_1.OP_r10_max', 'S0T302_3.DACA.PV_r15_max', 'S0CH4_1.DACA.PV_df1', 'S0FC301.PIDA.SP_df1', 'S0FC301.PIDA.SP_df2', 'S0T301_14.DACA.PV_df1', 'S0FC301.PIDA.PV_df1', 'S0FC301.PIDA.SP_df3', 'S0T301_14.DACA.PV_df2', 'S0P306_2.DACA.PV_df2', 'S0HV302_1.S0FV302_1.OP_r5_max', 'S0FC302.PIDA.PV_df2', 'S0FC302.PIDA.PV_df3', 'S0FC301.PIDA.SP_df5', 'S0P306_2.DACA.PV_df3'],
    'S0F313.DACA.PV': ['S0FC302.PIDA.PV_r10_mean', 'S0FC303.PIDA.PV_r5_mean', 'S0FC303.PIDA.PV_df2', 'S0PC020.PIDA.OP_r10_max', 'S0T301_23.DACA.PV_r5_mean', 'S0RFC316.CTLALGO.PV_r5_mean', 'S0FC303.PIDA.PV_r15_mean', 'S0FC301.PIDA.OP_r10_mean', 'S0FC303.PIDA.PV_df5', 'S0FK318.DACA.PV_r10_max', 'S0PC020.PIDA.OP_r10_mean', 'S0LC303.PIDA.PV_r5_min', 'S0FC302.PIDA.PV_df5', 'S0RFC322.PIDA.PV_r15_max', 'S0FC303.PIDA.PV_r15_max', 'S0FC303.PIDA.PV_r15_min'],
    'S0FK303.DACA.PV': ['S0FK204.DACA.PV_r15_max', 'S0PC020.PIDA.PV_df2', 'S0PC020.PIDA.OP_r10_max', 'S0T401_16.DACA.PV', 'S0T301_9.DACA.PV_df3', 'S0FC307.PIDA.PV_df2', 'S0FC307.PIDA.PV_r15_min', 'S0T301_24.DACA.PV_df5', 'S0T301_9.DACA.PV_df1', 'S0LC301.PIDA.PV_df4', 'S0PC020.PIDA.OP_r5_max', 'S0F451.DACA.PV_r15_min', 'S0FC301.PIDA.OP', 'S0FK318.DACA.PV_r10_min', 'S0T401_16.DACA.PV_r10_max', 'S0LC303.PIDA.PV_df1'],
    'S0T301_26.DACA.PV': ['S0T301_9.DACA.PV', 'S0TC402.PIDA.PV_r15_min', 'S0T301_23.DACA.PV', 'S0HV302_1.S0FV302_1.OP_r15_min', 'S0PC020.PIDA.OP_r10_min', 'S0RFC322.PIDA.PV_r15_min', 'S0T401_16.DACA.PV', 'S0T401_11.DACA.PV_r15_min', 'S0RFC322.PIDA.PV_r15_max', 'S0T301_23.DACA.PV_r15_min', 'S0PC020.PIDA.OP_r5_mean', 'S0PC020.PIDA.PV', 'S0RFC322.PIDA.PV_r10_mean', 'S0T301_24.DACA.PV_r15_max', 'S0FC303.PIDA.PV_r10_min', 'S0T301_9.DACA.PV_r15_max']
}