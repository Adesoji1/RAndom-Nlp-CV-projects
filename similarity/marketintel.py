"""
This script uses the requests and spacy libraries to fetch articles related to market intelligence from a news API and extract the entities and their labels from the article text. The entities are then printed to the console, allowing you to quickly gain insights into the latest market intelligence news.
"""

import requests
import spacy

# load the spacy model
nlp = spacy.load('en_core_web_sm')

# set the URL for the news API
url = 'https://newsapi.org/v2/everything?q=market+intelligence&apiKey=<your_api_key>'

# make a request to the news API
response = requests.get(url)

# extract the articles from the response
articles = response.json()['articles']

# loop through the articles
for article in articles:
    # parse the article text using spacy
    doc = nlp(article['description'])

    # print the entities and their labels
    print('Entities:')
    for ent in doc.ents:
        print(' - %s (%s)' % (ent.text, ent.label_))

    # print a separator
    print('-' * 20)
