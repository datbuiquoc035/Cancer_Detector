import numpy as np
from sklearn.preprocessing import LabelEncoder

NUMERIC_FEATURES = ['VariationID', 'CHR_GRCh38', 'Start_GRCh38', 'Stop_GRCh38']
CATEGORICAL_FEATURES = ['VariationType', 'GeneSymbol']
FEATURE_COUNT = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + 2

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['Ref_Allele', 'Alt_Allele', 'Clinical_Significance']


def _to_float(value, fallback=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback

def preprocess_variant(chromosome, position, gene='UNKNOWN', variant_type='SNP',
                       ref_allele='', alt_allele='', variation_id=0,
                       chr_grch38=0, start_grch38=0, stop_grch38=0,
                       label_encoders=None):
    features = []
    features.append(_to_float(variation_id, 0.0))
    features.append(_to_float(chr_grch38, 0.0))
    features.append(_to_float(start_grch38, float(position)))
    features.append(_to_float(stop_grch38, float(position)))
    if label_encoders and 'VariationType' in label_encoders:
        try:
            vt_encoded = label_encoders['VariationType'].transform([str(variant_type)])[0]
        except (ValueError, KeyError):
            vt_encoded = 0
    else:
        vt_encoded = 0
    features.append(float(vt_encoded))
    if label_encoders and 'GeneSymbol' in label_encoders:
        try:
            gene_encoded = label_encoders['GeneSymbol'].transform([str(gene)])[0]
        except (ValueError, KeyError):
            gene_encoded = 0
    else:
        gene_encoded = 0
    features.append(float(gene_encoded))
    ref_len = len(str(ref_allele)) if ref_allele else 0
    alt_len = len(str(alt_allele)) if alt_allele else 0
    features.append(float(ref_len))
    features.append(float(alt_len))
    return np.array(features).reshape(1, -1)

def extract_features_from_df(df, label_encoders=None, fit_encoders=False):
    feature_arrays = []
    feature_names = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            feature_arrays.append(df[col].fillna(0).astype(float).values.reshape(-1, 1))
        else:
            feature_arrays.append(np.zeros((len(df), 1)))
        feature_names.append(col)
    for col in CATEGORICAL_FEATURES:
        series = df[col].fillna('Unknown').astype(str)
        if fit_encoders:
            le = LabelEncoder()
            encoded = le.fit_transform(series)
            label_encoders[col] = le
        else:
            le = label_encoders.get(col) if label_encoders else None
            if le:
                unseen_mask = ~series.isin(le.classes_)
                encoded = le.transform(series[~unseen_mask])
                result = pd.Series(0, index=series.index, dtype=int)
                result[~unseen_mask] = encoded
                encoded = result.values
            else:
                encoded = np.zeros(len(df), dtype=int)
        feature_arrays.append(np.array(encoded, dtype=float).reshape(-1, 1))
        feature_names.append(col)
    ref_len = df['Ref_Allele'].astype(str).str.len().fillna(0).values.reshape(-1, 1)
    alt_len = df['Alt_Allele'].astype(str).str.len().fillna(0).values.reshape(-1, 1)
    feature_arrays.append(ref_len)
    feature_arrays.append(alt_len)
    feature_names.extend(['ref_len', 'alt_len'])
    X = np.hstack(feature_arrays)
    return X.astype(np.float32), feature_names
