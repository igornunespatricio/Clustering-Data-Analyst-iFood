def get_outliers(df, column):
    """Return outliers from a given column in a given dataframe

    Args:
        df (pd.DataFrame): a pandas dataframe containing the data to return the outliers
        column (string): the label of the column to search outliers for

    Returns:
        pd.DataFrame: subset of rows from df containing outliers from column
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Define outlier criteria
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers