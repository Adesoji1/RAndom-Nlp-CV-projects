from flask import Flask, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']
    
    # compute cosine similarity between text1 and text2
    similarity = cosine_similarity([text1], [text2])
    
    return similarity
    
if __name__ == '__main__':
    app.run()
