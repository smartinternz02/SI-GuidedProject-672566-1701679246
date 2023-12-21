from flask import Flask, render_template, request
import numpy as nd
import joblib

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')

@app.route('/')
@app.route("/home",methods=["GET","POST"])
def home():
    return render_template('home.html')
@app.route('/index',methods=["GET","POST"])
def index():
    return render_template('index.html')

@app.route('/adapt',methods=["GET","POST"])
def adapt():
    return render_template('adaptivity.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    gender = request.form['gender']
    age_range = request.form['age']
    class_duration_range = request.form['class_duration']
    financial_condition = request.form['financial_condition']
    institute_type = request.form['institute_type']
    network_type = request.form['network_type']
    education_level = request.form['education_level']
    location = request.form['location']
    internet_type = request.form['internet_type']
    it_student = request.form['it_student']
    device = request.form['device']
    self_lms = request.form['self_lms']
    load_shedding = request.form['load_shedding']

    # Encode categorical variables
    gender = 1 if gender == 'girl' else 0
    age_range = age_range.split('-')
    age = (int(age_range[0]) + int(age_range[1])) / 2
    class_duration_range = class_duration_range.split('-')
    class_duration = (int(class_duration_range[0]) + int(class_duration_range[1])) / 2
    financial_condition = 2 if financial_condition == 'poor' else (1 if financial_condition == 'mild' else 0)
    institute_type = 1 if institute_type == 'Government' else 0
    network_type = 2 if network_type == '2G' else (1 if network_type == '3G' else 0)
    education_level = 2 if education_level == 'college' else (1 if education_level == 'university' else 0)
    location = 1 if location == 'yes' else 0
    internet_type = 1 if internet_type == 'Wi-fi' else 0
    it_student = 1 if it_student == 'yes' else 0
    device = 2 if device == 'tab' else (1 if device == 'computer' else 0)
    self_lms = 1 if self_lms == 'yes' else 0
    load_shedding = 1 if load_shedding == 'high' else 0

    # Create input array
    input_data = nd.array([gender, age, class_duration, financial_condition, institute_type, network_type, education_level,
                            location, internet_type, it_student, device, self_lms, load_shedding])
    
    input_data = input_data.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Map prediction to adaptivity level
    adaptivity_level = 'High' if prediction == 2 else ('Moderate' if prediction == 1 else 'Low')

    return render_template('result.html', prediction=adaptivity_level)

if __name__ == '__main__':
    app.run(debug=True)
