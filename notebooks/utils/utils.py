import pandas as pd
import itertools
from scipy.stats import shapiro
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def best_transformations(dataframe, columns, transformations_dict):
    """
    Return the best transformation from transformations_dict for each column in a given dataframe. 
    The best transformation is determined by the pvalue of the shapiro normality test.
    """

    loop_this = itertools.product(columns, transformations_dict.keys())
    
    recommendations_column_names = []
    recommendations_transformation_names = []
    recommendations_pvalues = []

    for column, name in loop_this:

        transf_col = transformations_dict[name].fit_transform(dataframe[column].values.reshape(-1,1)) if transformations_dict[name] else dataframe[column]
        recommendations_column_names.append(column)
        recommendations_transformation_names.append(name)
        recommendations_pvalues.append(shapiro(transf_col).pvalue)

    recommendations = pd.DataFrame(
        {
            'Column': recommendations_column_names,
            'Transformation': recommendations_transformation_names,
            'pvalue_normality': recommendations_pvalues
        }
    )

    recommendations = recommendations.groupby('Column').apply(lambda x: x.sort_values(by='pvalue_normality', ascending=False)).reset_index(drop=True)
    
    return recommendations


def elbow_and_silhouete_charts(dataframe, random_state=42, range_k=(2,11)):
    """Plot Elbow and silhouette scores for the range_k values.

    Args:
        dataframe (pd.DataFrame): pandas dataframe with the data
        random_state (int, optional): Defaults to 42.
        range_k (tuple, optional): Range of k form KMeans to compute the elbow and silhouette scores. Defaults to (2,11).
    """

    elbow = {}
    silhouette = []
    k_range = range(*range_k)

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        # Fit the data
        kmeans.fit(dataframe)
        # Compute elbow score
        elbow[i] = (kmeans.inertia_)
        # Compute silhouette score
        silhouette.append(silhouette_score(dataframe, kmeans.labels_))

    fig, axs= plt.subplots(nrows=1, ncols=2, figsize=(15,5), tight_layout=True)

    # Elbow plot
    axs[0].plot(k_range, elbow.values(), marker='o')
    axs[0].set_xlabel('Number of clusters')
    axs[0].set_ylabel('Within-cluster sum of squares (Inertia)')
    axs[0].set_title('Elbow Method')

    # Silhouette plot
    axs[1].plot(k_range, silhouette, marker='o')
    axs[1].set_xlabel('Number of clusters')
    axs[1].set_ylabel('Silhouette score')
    axs[1].set_title('Silhouette Score')

    return fig