import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    print('Correlation between SDA and DDA FoG values')

    # Fetch the csv file with the DDA FoG data
    data_DDA = pd.read_csv('C:/Data/SDA_PICS/SDAvsDDA diskimageR results.csv')

    print(f'data_DDA.shape = {data_DDA.shape}')

    #df_ypd_48hr = data_DDA.loc[(data_DDA['Medium'] == 'YPD') & (data_DDA['Time'] == 48)]
    
    df_sdc_DDA_48hr = data_DDA.loc[(data_DDA['Medium'] == 'SDC') & (data_DDA['Time'] == 48)]

    print(f'df_ypd.shape = {df_sdc_DDA_48hr.shape}')
    print('df_sdc_48hr')
    print(df_sdc_DDA_48hr)

    print()
    
    # Fetch the SDA data file for SDC
    df_sdc_SDA_48hr = pd.read_excel('C:/Data/SDA_PICS/sdc_ISO_PL_1_summary_data.xlsx')

    

    # Prep data for scatter
    FoG_50_scd_SDA = list(df_sdc_SDA_48hr['FoG_50'])
    print('SDA FoG values')
    print(FoG_50_scd_SDA)

    FoG_50_sdc_DDA = list(df_sdc_DDA_48hr['FoG50'])
    
    print('DDA FoG values')
    print(FoG_50_sdc_DDA)

    # Create the scatter plot
    fig, ax = plt.subplot()

    


if __name__ == '__main__':
    main()