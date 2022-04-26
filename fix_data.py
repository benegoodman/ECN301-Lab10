# ----------------------------------------------------------
# -- fix_data.py                                          --
# --     Merge the CPI and GDP series                     --
# --     Create a few new variables                       --
# --     Create a few new variables                       --
# --     Olvar Bergland, sept 2018                        --
# ----------------------------------------------------------

import numpy  as np
import pandas as pd
import sys

#
# read data from csv file
dcpi = pd.read_csv('./data_raw/nz_cpi.csv',parse_dates=[0],index_col=0)
dgdp = pd.read_csv('./data_raw/nz_gdp.csv',parse_dates=[0],index_col=0)

#
# merge files
df = dcpi.join(dgdp)

#
# drop a few nan
df = df.dropna()

#
# quarterly data
# df.index = df.index.to_period('Q')
# df['quarter'] = df.index.quarter
#

#
# ensure this is absolutely in the right order
df = df.sort_index().copy()

#
# add trend
df['trend'] = np.arange(len(df))/4

#
print(df.index[0])
print(df.index[-1])
print(df.head())
print(df.info())


#
df.to_csv('./data/nz_gdp.csv')
