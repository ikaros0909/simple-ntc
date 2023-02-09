from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.get_json()
    p_data = [{"model_fn": ".\models\y.native.kcbert.pth"}
        , {"train_fn": ".\data\y_test.tsv"}
        , {"gpu_id": 0}
        , {"top_n": 10}]
    # TODO: classify_plm.py 모델을 사용하여 예측을 수행하는 로직 구현
    result = subprocess.run(['python', '.\predict.py', '--model_fn', '.\models\y.native.kcbert.pth', '--test_file', '.\data\y_test.tsv', '--gpu_id', '0', '--top_n', '20'], stdout=subprocess.PIPE)

    # return str(result.stdout.decode())
    return result.stdout.decode('euc-kr')

    # result = {"prediction": 42}
    # return result

if __name__ == "__main__":
    app.run(debug=True, host='10.1.0.61', port=80)
