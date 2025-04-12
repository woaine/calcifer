import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the first two rows and reset index
    df.columns = df.iloc[1]
    df = df[2:].reset_index(drop=True)

    # Keep only specified columns
    df = df[[  
        "T_offset1", "T_offset2", "T_offset3", "T_offset4",
        "T_FHBC1", "T_FHBC2", "T_FHBC3", "T_FHBC4",
        "aveOralM", "T_atm"
    ]]

    # Normalize the thermal readings
    for i in range(1, 5):
        # Deduct T_offset* values from T_FHBC* to correct for sensor bias or environmental drift
        df[f'T_FHBC{i}'] = pd.to_numeric(df[f'T_FHBC{i}'], errors='coerce') - pd.to_numeric(df[f'T_offset{i}'], errors='coerce')
        df.drop(columns=[f'T_offset{i}'], inplace=True)

    # Average the values of all T_FHBC* and assign to a new column Tg
    df['Tg'] = df[[f'T_FHBC{i}' for i in range(1, 5)]].astype(float).mean(axis=1)

    # Drop all T_FHBC* columns
    df.drop(columns=[f'T_FHBC{i}' for i in range(1, 5)], inplace=True)

    # Rename columns
    df.rename(columns={'aveOralM': 'Tc', 'T_atm': 'Ta'}, inplace=True)

    # Reorder columns
    return df[['Tg', 'Ta', 'Tc']]

def save_clean_data_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)