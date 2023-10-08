import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    print('Let\'s fucking go!')

    # Fetch the csv file with the DDA FoG data
    data_DDA = pd.read_csv('C:/Data/SDA_PICS/SDAvsDDA diskimageR results.csv')

    df_ypd_48hr = data_DDA.loc[(data_DDA['Medium'] == 'YPD') & (data_DDA['Time'] == '48')]
    
    df_sdc_48hr = data_DDA.loc[(data_DDA['Medium'] == 'SDC') & (data_DDA['Time'] == '48')]

    print(df_ypd_48hr)

if __name__ == '__main__':
    main()