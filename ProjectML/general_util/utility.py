def my_l_concat(df1, df2):
    import pandas as pd

    df1.reset_index(inplace=True,drop=True)
    df2.reset_index(inplace=True,drop=True)
    df_result = pd.concat([df1,df2],axis=1)
    return df_result