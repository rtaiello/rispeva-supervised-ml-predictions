# third part
import numpy as np

def drop_corr_feature(X):
    corr_matrix = X.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.7)]
    X = X.drop(columns=to_drop,axis=1)
    return X