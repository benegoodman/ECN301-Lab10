# --------------------------------------------------------------
# - GET_DATA                                                   -
# -                                                            -
# - Get data from db*nomics using their API framework          -
# - Raw datafiles are stored as CSV-files                      -
# -                                                            -
# - Olvar Bergland  jan 2021                                   -
# --------------------------------------------------------------


import sys
import pandas as pd
import os

os.chdir('yourpath')


from dbnomics import fetch_series, fetch_series_by_api_link


def get_cpi():

    # --------------------------------------------------------------------
    # - CPI
    # - Consumer Price Index > All items
    # - NZL.CPALTT01.IXNB.Q
    # - https://db.nomics.world/OECD/MEI/NZL.CPALTT01.IXNB.Q
    # - Relevant variables: period and value
    # --------------------------------------------------------------------

    dfx = fetch_series('OECD/MEI/NZL.CPALTT01.IXNB.Q')
    dfx = dfx.rename(columns={'value': 'cpi'})
    print(dfx[['period','cpi']])

    return dfx[['period','cpi']]


def get_gdp():

    # --------------------------------------------------------------------
    # - Gross Domestic Product (GDP)
    # - Index 2015=100
    # - Seasonally adjusted
    # - OECD/MEI/NZL.LORSGPOR.IXOBSA.Q
    # - https://db.nomics.world/OECD/MEI/NZL.LORSGPOR.IXOBSA.Q
    # - Relevant variables: period and value
    # --------------------------------------------------------------------
    #   https://db.nomics.world/OECD/MEI/NZL.NAEXCP01.STSA.Q
    #   https://db.nomics.world/OECD/MEI/NZL.NAGIGP01.IXOBSA.Q
    # --------------------------------------------------------------------
    #

    dfx = fetch_series('OECD/MEI/NZL.LORSGPOR.IXOBSA.Q')
    dfx = dfx.rename(columns={'value': 'gdp'})
    print(dfx[['period','gdp']])

    return dfx[['period','gdp']]

#%%

if __name__ == "__main__":

    if len(sys.argv) == 1:

        #
        # CPI
        dfcpi = get_cpi()
        dfcpi = dfcpi.set_index('period').sort_index()
        dfcpi.to_csv('./data_raw/nz_cpi.csv')

        #
        # GDP
        df_gdp = get_gdp()
        df_gdp = df_gdp.set_index('period').sort_index()
        df_gdp.to_csv('./data_raw/nz_gdp.csv')

    else:

        print('Usage: python get_data.py')
