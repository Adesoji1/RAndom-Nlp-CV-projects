import requests

url = 'http://localhost:5000/similarity'
data = {'text1': 'This is a test', 'text2': 'This is also a test'}

response = requests.post(url, json=data)
print(response.json())
