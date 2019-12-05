# import external dependencies
import numpy as np
import pandas as pd
import pdb

def fix_rental(df):

    print("\nfixing rental data...")

    df['lives_at_res'] = ( (df.address == df.situsaddress) | df.address.isnull() ).astype(np.int)
    df['lives_in_city'] = ( (df.address != df.situsaddress) & (df.city == 'TUCSON') ).astype(np.int)
    df['lives_in_state'] = ( (df.address != df.situsaddress) & (df.city != 'TUCSON') & (df.state == 'AZ') ).astype(np.int)
    df['lives_out_of_state'] = ( (df.address != df.situsaddress) & (df.state != 'AZ') & (df.address.notna()) ).astype(np.int)

    # drop the original columns
    df.drop(columns=['city', 'state', 'address', 'situsaddress'], inplace=True)
