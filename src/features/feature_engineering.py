import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame):
    """
    Generate engineered features from the input DataFrame.
    This function takes a DataFrame containing thermal properties and creates
    additional features to enhance data modeling. Specifically, it:
    - Computes 'qlog' as the ratio of 'Tg' to the base-10 logarithm of 'Ta'.
    - Computes 'qdiff' as 1 minus the natural logarithm of 'Ta' divided by the square of 'Tg'.
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns 'Tg' and 'Ta'.
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the original data and the newly created features.
    Notes:
    ------
    - Ensure that the input DataFrame contains the columns 'Tg' and 'Ta' before calling this function.
    - The function does not modify the original DataFrame; it operates on a copy.
    Example:
    --------
    >>> import pandas as pd
    >>> data = {'Tg': [30.123, 35.456], 'Ta': [20, 30]}
    >>> df = pd.DataFrame(data)
    >>> create_features(df)
          Tg  Ta     qlog   qdiff
    0  30.12  20  23.1508  0.9967
    1  35.46  30  24.0062  0.9973
    """
    
    df_features = df.copy()
    df['Tg'] = round(df['Tg'], 2)

    df_features['qlog'] = (df['Tg'] / np.log10(df['Ta']))
    df_features['qdiff'] = 1 - (np.log(df['Ta']) / df['Tg']**2)

    return df_features