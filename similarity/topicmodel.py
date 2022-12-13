"""
create Topic modelling using Kmeans clustering to group customer reviews
This script uses the pandas and sklearn libraries to group customer reviews into topics using K-means clustering. The reviews are first represented as a tf-idf matrix, which is then used to fit the K-means model. The reviews are then assigned to the clusters that the model predicts. The first 10 reviews and their cluster assignments are printed to the console.

"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# read in the customer reviews
reviews = pd.read_csv('customer_reviews.csv')

# create the tf-idf matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews['text'])

# create the K-means model
kmeans = KMeans(n_clusters=5)

# fit the model to the tf-idf matrix
kmeans.fit(tfidf_matrix)

# assign each review to a cluster
reviews['cluster'] = kmeans.predict(tfidf_matrix)

# print the first 10 reviews and their cluster assignments
print(reviews[['text', 'cluster']].head(10))
