# --------------------------------------------------------------
# - GDP_MODEL                                                  -
# -                                                            -
# - Estimation and forecasting with a VAR(x) model             -
# - Out-of sample comparison of forecast ability               -
# -                                                            -
# - Olvar Bergland  feb 2019                                   -
# --------------------------------------------------------------


#
#
import sys
import os
import math
from datetime import datetime

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

#%%

# --------------------------------------------------------------------

#
def get_data():
    #
    #
    #
    fn_data = './data/nz_gdp.csv'

    # load data
    if os.path.isfile(fn_data):
        df = pd.read_csv(fn_data,parse_dates=[0],index_col=[0])
        df = df[df.index<'2020-01-01']
        df.index = df.index.to_period('Q')
        df = df.sort_index()
    else:
        sys.exit('Could not open load data file {}'.format(fn_data))

    df['lgdp'] = 100*np.log(df.gdp)
    df['lcpi'] = np.log(df.cpi)
    df['inf' ] = 400*(df.lcpi - df.lcpi.shift(1))

    df['dgdp'] = df.lgdp - df.lgdp.shift(1)
    df['dinf'] = df.inf  - df.inf.shift(1)

    return df

#%%
# --------------------------------------------------------------------

def plot_level(df):
    #
    # plot log GDP series
    #

    fig, ax = plt.subplots(2, figsize=(6,8))
    df['lgdp'].plot(ax=ax[0])
    ax[0].set_ylabel('100 * Log GDP (2015=100)')
    df['inf'].plot(ax=ax[1])
    ax[1].set_ylabel('Annual infaltion (%)')
    ax[1].set_xlabel('Time')
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig('./figs/nz_lgdp.png')
    plt.close()

#%%
# --------------------------------------------------------------------

def plot_diff(df):
    #
    # plot diff of GDP series
    #

    fig, ax = plt.subplots(2, figsize=(6,8))
    df['dgdp'].plot(ax=ax[0])
    ax[0].set_ylabel('Change in Log GDP')
    df['dinf'].plot(ax=ax[1])
    ax[1].set_ylabel('Change in Annual Infaltion')
    ax[1].set_xlabel('Time')
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig('./figs/nz_dgdp.png')
    plt.close()


#%%
# --------------------------------------------------------------------

def plot_fcast(df):
    #
    # plot diff of GDP series
    #

    fig, ax = plt.subplots(2, figsize=(6,8))
    df['lgdp'].plot(ax=ax[0])
    df['lghat'].plot(ax=ax[0])
    ax[0].set_ylabel('Log GDP')
    df['inf'].plot(ax=ax[1])
    df['lihat'].plot(ax=ax[1])
    ax[1].set_ylabel('Annual Infaltion')
    ax[1].set_xlabel('Time')
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()
    fig.show()
    fig.savefig('./figs/nz_fcast.png')
    

#%%



#%%
# --------------------------------------------------------------------

def var_model(df):
    #
    # estimate the VAR model
    #

    #
    # create variables
    df['dgdp_L1'] = df.dgdp.shift(1)
    df['dgdp_L2'] = df.dgdp.shift(2)
    df['dgdp_L3'] = df.dgdp.shift(3)
    df['dgdp_L4'] = df.dgdp.shift(4)
    df['dinf_L1'] = df.dinf.shift(1)
    df['dinf_L2'] = df.dinf.shift(2)
    df['dinf_L3'] = df.dinf.shift(3)
    df['dinf_L4'] = df.dinf.shift(4)

    #
    # estimation sample
    df_est = df[(df.index >= pd.Period('1963Q1')) & (df.index <= pd.Period('2003Q4'))].copy()
    df_prd = df[(df.index >= pd.Period('2003Q1')) & (df.index <= pd.Period('2007Q4'))].copy()

    #
    # estimate VAR style model
    #
    
    # HAC = heteroskedasticity-autocorrelation robust covariance
    # We use 4 lags because we have 4 lagged variables
    var_gdp = smf.ols(formula='dgdp ~ dgdp_L1 + dgdp_L2 + dgdp_L3 + dgdp_L4 + dinf_L1 + dinf_L2 + dinf_L3 + dinf_L4',
                      data=df).fit().get_robustcov_results(cov_type='HAC', maxlags=4) 
    print(var_gdp.summary())

    var_inf = smf.ols(formula='dinf ~ dinf_L1 + dinf_L2 + dinf_L3 + dinf_L4 + dgdp_L1 + dgdp_L2 + dgdp_L3 + dgdp_L4',
                      data=df).fit().get_robustcov_results(cov_type='HAC', maxlags=4)
    print(var_inf.summary())

    #
    # create forecast
    #  dynamic forecast starting in 2006
    #  with four lags need 2005 data as well
    #

    #
    # keep forecast here
    lghat = np.zeros(len(df_prd))
    dghat = np.zeros(len(df_prd))
    lihat = np.zeros(len(df_prd))
    dihat = np.zeros(len(df_prd))

    #
    # loop over obs in the forecast dataset
    t = 0
    for q, v in df_prd.iterrows():
        if q < pd.Period('2004Q1'):
            dghat[t] = v['dgdp']
            lghat[t] = v['lgdp']
            dihat[t] = v['dinf']
            lihat[t] = v['inf' ]
        else:
            #
            # GDP forecast
            dghat[t] = (var_gdp.params[0] +
                        var_gdp.params[1]*dghat[t-1] +
                        var_gdp.params[2]*dghat[t-2] +
                        var_gdp.params[3]*dghat[t-3] +
                        var_gdp.params[4]*dghat[t-4] +
                        var_gdp.params[5]*dihat[t-1] +
                        var_gdp.params[6]*dihat[t-2] +
                        var_gdp.params[7]*dihat[t-3] +
                        var_gdp.params[8]*dihat[t-4] )
            lghat[t] = lghat[t-1] + dghat[t]

            #
            # INF forecast
            dihat[t] = (var_inf.params[0] +
                        var_inf.params[1]*dihat[t-1] +
                        var_inf.params[2]*dihat[t-2] +
                        var_inf.params[3]*dihat[t-3] +
                        var_inf.params[4]*dihat[t-4] +
                        var_inf.params[5]*dghat[t-1] +
                        var_inf.params[6]*dghat[t-2] +
                        var_inf.params[7]*dghat[t-3] +
                        var_inf.params[8]*dghat[t-4])
            lihat[t] = lihat[t-1] + dihat[t]

        #
        t += 1

    #
    # keep forecast
    df_prd['lghat'] =  lghat
    df_prd['lihat'] =  lihat

    #
    # plot forecast
    plot_fcast(df_prd)

#%%
# --------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) == 1:

        #
        # get data
        df = get_data()
        #print(df.tail())

        #
        # plot data
        plot_level(df)
        plot_diff(df)
        
        #
        # test for stationarity
        #  not included

        #
        # VAR model: estimation and out of sample testing
        var_model(df)


    else:

        print('Usage: python get_data.py')
