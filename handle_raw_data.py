import pandas as pd
import numpy as np

def trim_data():

    #===========================================================================
    # eco district
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/EDNPI19.csv', low_memory=False)

    # select useful columns
    data = data[['PARCEL', 'LAT', 'LON', 'LANDSQFT']]
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/eco.csv', index=False)

    #===========================================================================
    # notice of value
    #===========================================================================

    # load data
    data = pd.read_csv(f"data/raw_data/Notice19.csv", sep=',', encoding='latin', dtype='str', low_memory=False)
    for col in data: data[col].astype(int, errors='ignore')

    # select useful columns
    data = data[['PARCEL', 'LANDFCV']] #ZIP
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/notice.csv', index=False)

    #===========================================================================
    # parcel model data -- just get latest set
    #===========================================================================

    # load data
    data = pd.read_csv('data/raw_data/PclModelData19.csv', low_memory=False)

    # select useful columns
    data = data[['Parcel', 'Age']]
    data = data.rename(str.lower, axis='columns')

    # save
    data.to_csv('data/trimmed/parcel.csv', index=False)

def join_trimmed_data():

    # get starting DataFrame to join to -- pick smallest one
    JOIN = pd.read_csv("data/clean_residential_and_sale.csv", low_memory=False, index_col="Unnamed: 0")
    # rename columns to lower case
    JOIN = JOIN.rename(str.lower, axis='columns')

    # make a function to join to JOIN with filename as input
    def join_it(JOIN,file):

        print(f"\nmerging {file}")

        # get the full file name
        filename = f'data/trimmed/{file}.csv'

        # load the dataframe
        df = pd.read_csv(filename, low_memory=False)
        # # rename columns to lower case
        # df = df.rename(str.lower, axis='columns')

        # perform inner join
        JOIN = JOIN.merge(df, how='inner', on='parcel')

        # delete df
        del df

        # output
        return JOIN

    # go through and join_it for each of the remaning dataframes
    JOIN = join_it(JOIN,'eco')
    JOIN = join_it(JOIN,'notice')
    JOIN = join_it(JOIN,'parcel')

    # save
    print("\nsaving as 'data/dirty.csv'")
    JOIN.to_csv("data/dirty.csv", index=False)

def clean_and_transform():

    # load data
    data = pd.read_csv("data/dirty.csv", low_memory=False)

    # drop columns
    data.drop(columns=['landfcv', 'age'], inplace=True)

    # # one-hot data
    # one_hot_columns = ['zip']
    # for col in one_hot_columns:
    #     uniques = data[col].unique()
    #     for u in uniques:
    #         data[f"{col}_{u}"] = (data[col] == u).astype(np.int32)
    # data.drop(columns=one_hot_columns, inplace=True)

    # drop columns
    drop_columns=['parcel', 'heat_2.0', 'validationdescription_buyer/seller is a non-profit institution', 'validationdescription_payoff of land contract']
    data.drop(columns=drop_columns, inplace=True)

    # shuffle the entire set of observations
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    # get the numbers of observations for the data sets
    N = data.shape[0]
    N_train = int(.6*N)
    N_validate = int(.2*N)
    N_test = N - N_validate - N_train

    # get the cross validation sets
    train = data[ :N_train ]
    validate = data[N_train : N_train+N_validate ]
    test = data[ N_train+N_validate: ]

    # save the dataframes
    train.to_csv("data_sets/train.csv", index=False)
    validate.to_csv("data_sets/validate.csv", index=False)
    test.to_csv("data_sets/test.csv", index=False)
