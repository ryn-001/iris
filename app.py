from flask import Flask, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def home():
    # Read the index.html file and return it
    with open('index.html') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                               columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    prediction = loaded_model.predict(input_data)

    # Read the result.html file and render it with the prediction
    with open('result.html') as f:
        result_html = f.read()
    result_html = result_html.replace('{{ prediction }}', prediction[0])  # Replace placeholder

    return result_html

if __name__ == '__main__':
    app.run(debug=True)
