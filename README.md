# Ifood Case

iFood is a Brazilian online food ordering and food delivery platform, operating in Brazil and Colombia. 

This project was a case developed with the goal to hire a data analyst/data scientist.

The main goals of this case are to perform exploratory data analysis, clustering and classification models. More details can be seen in the pdf in the case folder.

## Project Structure

The project is separated in the following folders:

- case: contains the pdf of the case with more detailed information about the tasks to be accomplished
- config: a configuration folder for storing configuration files like the yaml file for the ydata profiling report
- data: a folder storing the raw data (ml_project1_adata.csv), the cleaned data produced from the eda notebook and the customers_clustered.csv performed after clustering
- images: contains project images to be displayed
- notebooks: contains notebooks used for exploratory data analysis, clustering an, classification and hyperparameter tuning
- reports: the report containing the html file generated from the ydata profiling report

## Customer Segmentation

The segmentation part aims to categorize customers into distinct gtoups based on their caracteristics provided in the dataframe. This categorization enables target marketing strategies and more personalized customer experiences.

For this segmentation, a pipeline using scikit learn was created to preprocess the data using standard scalers, power transformations and min-max scalers, this data was fed to a KMeans clustering algorithm and the resulting analysis for the 3 clusters can be seen below.

![clustering boxplots](/images/cluster_comparing.png)

- Cluster 1 is represented by small income people relatively to clusters 0 and 2. They hold the smallest amount purchased when compared to the other clusters for all product categories: fish, meat, wines, etc. The same behaviour can be seen with number of purchases, except for the deals purchases where cluster 1 contains on average more deals purchased then cluster 0. The age doesn't change much between clusters, but looking at the median values we can say that cluster 1 is on average represented by younger people than clusters 0 and 2. Cluster 0 also represents those people that visits the website most often compared to the other clusters.

- Cluster 2 is represented by average income people relatively to cluster 0 and 1. The amount purchased and the number of purchases are generally on average as well, except for deal purchases and web purchases, where cluster 2 represents the biggest amount of deals purchases and its web purchases are comparable to cluster 0. When looking at the age, again there is no big separation, but looking at the median, we can say they are older than the other two clusters. They also visit the website more often compared to cluster 0, and almost equivalent to cluster 1.

- Cluster 0 is represented by the high income people relatively to clusters 1 and 2. The amount purchased by them is the highest for all product categories and they are also the ones with higher number of purchases with respect to web, catalog and store purchases, except for the number of deal purchases, where they are least frequent. They also don't visit the website much as compared to clusters 1 and 2.

## Classification

In the classification task, the goal is to predicct whether a customer will accept the campaign or not based on the dataset features.

For this task we utilized a pipeline built on top of scikit learn that again preprocess the data as done in the segmentation part, but also adds the two more steps for feature selection using ANOVA and random under sampler because of the imbalanced dataset with 15% acceptance rate. 3 models were trained (logistic regression, decision tree and k nearest neighbors) plus a dummy classifier as a baseline. The best model was considered to be the logistic regression, due to its superior performance in the average precision and roc auc metrics. 

![model comparision](/images/classification_model_comparision.png)

After selecting the logistic regression as the best model, we tuned its hyperparameters and performed analysis on the weights associated with each feature in the dataset. This resulted in the chart below, where we can see the most important features considered in the model. Positive and negative values indicate increase and decrease in chance to accept the campaign, respectively. We can see days since enrolled, recency, total accepted campaign and amount  spent in regular products being the top 4 features driving the chance of accepting the campaign.


![logistic regression weights](/images/logistic_regression_weights.png)
