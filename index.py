from flask import Flask, request, jsonify
from functools import wraps
import jwt
import subprocess
import json
from predict import main

app = Flask(__name__)

SECRET_KEY = "JinhakSolutions#2023"

# python index.py --port=80

def check_for_token(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        # Check if the authorization header is present
        if "Authorization" not in request.headers:
            return jsonify({"error": "Authorization header is missing"}), 401
        
        # Extract the token from the header
        token = request.headers["Authorization"].split(" ")[1]

        try:
            # Decode the token and verify its signature
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.DecodeError:
            return jsonify({"error": "Invalid token"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401

        # Call the decorated function and pass the decoded payload as an argument
        return func(decoded_token, *args, **kwargs)

    return wrapped

@app.route("/")
def hello():
    return "Hello World!"

# Route that generates a JWT token
@app.route("/login", methods=["POST"])
def login():
    if request.is_json:
        # Extract the username from the request body
        username = request.json.get("username", None)
    else:
        return jsonify({"error": "The request payload is not in JSON format"}), 400

    # Check if the username is present
    if not username:
        return jsonify({"error": "Username is missing"}), 400

    print(username)
    # Create a payload with the username
    payload = {"username": username}

    # Encode the payload into a JWT token
    # token = jwt.encode(payload, SECRET_KEY, algorithm="HS256").decode("utf-8")
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    return jsonify({"token": token})

@app.route("/predict", methods=["POST"])
@check_for_token
def predict(decoded_token):
    data = request.get_json()
    data_string = json.dumps(data).encode('utf-8')

    # TODO: classify_plm.py 모델을 사용하여 예측을 수행하는 로직 구현
    result = subprocess.run(['python', '.\predict.py', '--model_fn', '.\models\y.native.kcbert_20230228_1.pth', '--gpu_id', '0'] #gpu_id=0
        , input=data_string
        , stdout=subprocess.PIPE
        , stderr=subprocess.PIPE
        # , capture_output=True
        # , text=True
        # , encoding='euc-kr'
    )

    # print(result.stderr)
    output = result.stdout.decode('euc-kr')

    # output = main(json.dumps(data).encode('utf-8'))

    data = {"output": output}
    return json.dumps(data)

    # return str(result.stdout.decode())
    # return result.stdout.decode('euc-kr')

    # result = {"prediction": 42}
    # return result


if __name__ == "__main__":
    app.run(debug=True, host='10.1.0.61', port=80)
