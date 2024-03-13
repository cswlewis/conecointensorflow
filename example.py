import requests

url = 'http://localhost:5000/predict'
data = {
    "data": {
        "Open": 54123.2,
        "High": 54219.6,
        "Low": 54020.8,
        "Close": 54186.7,
        "Volume_BTC": 2.90480747
    }
}
response = requests.post(url, json=data)

print(response.json())