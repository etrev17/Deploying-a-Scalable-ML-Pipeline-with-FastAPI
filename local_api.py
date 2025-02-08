import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"
request = requests.get(url)

# TODO: print the status code
print(request.status_code)

# TODO: print the welcome message
print(request.text)



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
post_url = url+"/data"
finalmsg = requests.post(post_url, json=data)

# TODO: print the status code
print(finalmsg.status_code)

# TODO: print the result
print(finalmsg.text)