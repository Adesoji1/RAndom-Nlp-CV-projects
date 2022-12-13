"""
This script uses the flask, requests, and spacy libraries to create a RESTful API that provides market intelligence news articles. The API uses the spacy library to extract the entities and their labels from the article text, and returns the articles with their entities as a JSON object. You can access the API using an HTTP GET request and use the entities to quickly gain insights into the latest market intelligence news.
"""


from flask import Flask, request
import requests
import spacy

# set up the Flask app
app = Flask(__name__)

# load the spacy model
nlp = spacy.load('en_core_web_sm')

# define the route for the API
@app.route('/articles', methods=['GET'])
def articles():
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

        # extract the entities and their labels
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # add the entities to the article object
        article['entities'] = entities

    # return the articles as a JSON object
    return jsonify(articles)

# run the app
if __name__ == '__main__':
    app.run()
