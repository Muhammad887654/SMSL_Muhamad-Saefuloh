import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df_raw):
    """
    Performs automatic preprocessing on the raw DataFrame,
    returning features (X) and target (y) ready for training.
    """

    df = df_raw.copy() # Work on a copy to avoid modifying the original

    # 1. Handle Missing Values
    median_acreage = df['Acreage'].median()
    df['Acreage'].fillna(median_acreage, inplace=True)

    mode_education = df['Education'].mode()[0]
    df['Education'].fillna(mode_education, inplace=True)

    # 2. Hapus Data Duplikat
    df = df.drop_duplicates()

    # 3. Encoding Data Kategorikal (One-Hot Encoding)
    categorical_cols = df.select_dtypes(include='object').columns
    columns_to_exclude_from_encoding = ['County', 'Farmer', 'Crop', 'Power source', 'Water source', 'Crop insurance']
    columns_to_encode = [col for col in categorical_cols if col not in columns_to_exclude_from_encoding]
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True, dtype=int)

    # 4. Normalisasi Fitur Numerik (Min-Max Scaling)
    scaler = MinMaxScaler()
    numerical_cols_to_normalize = [
        'Household size', 'Acreage', 'Fertilizer amount', 'Laborers',
        'Yield', 'Latitude', 'Longitude'
    ]
    df_encoded[numerical_cols_to_normalize] = scaler.fit_transform(df_encoded[numerical_cols_to_normalize])

    # 5. Deteksi dan Penanganan Outlier (Capping dengan IQR)
    numerical_cols_for_outliers = [
        'Household size', 'Acreage', 'Fertilizer amount', 'Laborers',
        'Yield', 'Latitude', 'Longitude'
    ]
    for col in numerical_cols_for_outliers:
        Q1 = df_encoded[col].quantile(0.25)
        Q3 = df_encoded[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_encoded[col] = df_encoded[col].clip(lower=lower_bound, upper=upper_bound)

    # 6. Separate features (X) and target (y)
    y = df_encoded['Yield']

    columns_to_exclude_from_X = [
        'Yield', 'County', 'Farmer', 'Crop', 'Power source',
        'Water source', 'Crop insurance'
    ]
    X = df_encoded.drop(columns=columns_to_exclude_from_X)

    return X, y

if __name__ == '__main__':
    # Example usage (assuming 'corn_data.csv' is in the same directory)
    print("Running example usage of preprocess_data function...")
    raw_data = pd.read_csv('corn_data.csv')
    X_processed, y_processed = preprocess_data(raw_data.copy()) # Pass a copy to the function
    print("\nPreprocessed X head:")
    print(X_processed.head())
    print("\nPreprocessed Y head:")
    print(y_processed.head())
    print("\nShape of X_processed:", X_processed.shape)
    print("Shape of y_processed:", y_processed.shape)
