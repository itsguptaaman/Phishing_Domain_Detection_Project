from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask_cors import CORS, cross_origin
from storing_model import StoreModel
from training_model import Training
from database_operations import Database
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
@cross_origin()
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    one = request.form.get('1')
    two = request.form.get('2')
    three = request.form.get('3')
    four = request.form.get('4')
    five = request.form.get('5')
    six = request.form.get('6')
    seven = request.form.get('7')
    eight = request.form.get('8')
    # print(one, two, three, four, five, six, seven, eight)
    user_input = pd.DataFrame([[one, two, three, four, five, six, seven, eight]],
                              columns=['directory_length', 'qty_slash_url', 'qty_dot_directory', 'file_length',
                                       'qty_hyphen_directory', 'qty_percent_file', 'qty_hyphen_file',
                                       'qty_underline_directory'])

    read = StoreModel()
    k_means = read.read_model("k_means")
    cluster = k_means.predict(user_input)
    # print(cluster)
    scale = read.read_model("Scale")
    user_input = pd.DataFrame(scale.transform(user_input), columns=user_input.columns)
    if cluster[0] == 0:
        model = read.read_model("gradient_boosting_cluster0")
        prediction = model.predict(user_input)
        # print(prediction)
        if prediction[0] == 0:
            prediction = "The Domain is real..!"
        else:
            prediction = "The Domain is Fake..!"
    elif cluster[0] == 1:
        model = read.read_model("support_vector_classifier_cluster1")
        prediction = model.predict(user_input)
        # print(prediction)
        if prediction[0] == 0:
            prediction = "The Domain is real..!"
        else:
            prediction = "The Domain is Fake..!"

    return str(prediction)


@app.route('/train', methods=['GET', 'POST'])
@cross_origin()
def train():
    try:
        path = r"dataset_full.csv"
        # path=""
        df = pd.read_csv(path)
    except Exception as e:
        data = Database()
        df = data.fetch_data()

    train_me = Training(df)
    cluster_model = train_me.train_model()
    # print(cluster_model)
    # return jsonify(cluster_model)
    return render_template("index.html", cluster_model=cluster_model)


if __name__ == '__main__':
    app.run(debug=True, )
