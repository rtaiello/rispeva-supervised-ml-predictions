def my_l_concat_series(df1, series):
    import pandas as pd

    df1.reset_index(inplace=True, drop=True)
    series.reset_index(inplace=True, drop=True)
    df_result = pd.concat([df1, series], axis=1)
    return df_result


def my_l_concat_dataframe(df1, df2):
    import pandas as pd

    df1.reset_index(inplace=True, drop=True)
    df2.reset_index(inplace=True, drop=True)
    df_result = pd.concat([df1, df2], axis=0)
    return df_result
