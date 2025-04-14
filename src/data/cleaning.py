import pandas as pd

def clean_data(df:pd.DataFrame, data_group:str=None):
    # Remove the first row
    df = df.iloc[1:].reset_index(drop=True)
    # Make first row as headers
    df.columns = df.iloc[0]
    # Reset index
    df = df[1:].reset_index(drop=True)

    # Change column types to float64
    df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4', 'aveOralM', 'T_atm']] = df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4', 'aveOralM', 'T_atm']].astype('float64')

    # Calculate the average of T_FHBC1 to T_FHBC4
    df['Tg'] = df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4']].mean(axis=1)

    # Keep specified columns
    df = df[['Tg', 'T_atm', 'aveOralM']]

    # Rename columns
    df = df.rename(columns={'T_atm': 'Ta', 'aveOralM': 'Tc'})

    # Reorder columns
    return df[['Tg', 'Ta', 'Tc']]

def save_clean_data_to_csv(clean_data: pd.DataFrame, path: str):
    clean_data.to_csv(path, index=False)