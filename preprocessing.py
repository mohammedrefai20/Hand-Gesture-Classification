import pandas as pd 
import numpy as np 

def preprcessing(filepath:str):

    data=pd.read_csv(filepath)

    df=pd.DataFrame(data)

    for axis in ['x', 'y']:
        base_col = f'{axis}1'
        cols = [f'{axis}{i}' for i in range(2, 22)]
        df[cols] = df[cols].sub(df[base_col], axis=0)

    norm = np.sqrt(df['x13']**2 + df['y13']**2)
    norm = norm.replace(0, np.nan)
    X_col = [f'x{i}' for i in range(1, 22)]
    y_col = [f'y{i}' for i in range(1, 22)]
    df[X_col] = df[X_col].div(norm, axis=0)
    df[y_col] = df[y_col].div(norm, axis=0)

    X=df.drop(['label'],axis=1)
    y=df['label']

    return X,y

