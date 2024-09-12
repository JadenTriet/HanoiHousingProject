import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

dfinput = pd.read_csv(r"C:\Users\Jaden\Desktop\python\HanoiHousingPricePrediction\dfemp.csv")
app = Flask(__name__)
model = pickle.load(open(r"C:\Users\Jaden\Desktop\python\HanoiHousingPricePrediction\model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\Jaden\Desktop\python\HanoiHousingPricePrediction\scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    paperwork = request.form.get('paperwork')
    floors = request.form.get('floors')
    rooms = request.form.get('rooms')
    area = request.form.get('area')
    district = request.form.get('district')
    ward = request.form.get('ward')
    houseType = request.form.get('houseType')

    if any(value is None or value == '' for value in [floors, rooms, area, district, ward, houseType]):
        error_message = 'Error: Please fill in all the fields!'
        return render_template('index.html', error_message=error_message)
    
    try:
        floors = int(floors)
        rooms = int(rooms)
        area = float(area)
        paperwork = bool(paperwork)
    except ValueError:
        error_message = 'Error: Please fill in numerical values for appropriate fields'
        return render_template('index.html', error_message=error_message)

    if area > 100 or area < 10:
        error_message = 'Error: Please fill in a valid area value between 10 and 100!'
        return render_template('index.html', error_message=error_message)
    
    dfinputnew = dfinput.copy()
    dfnew=[0] * 283
    dfnew[0] = paperwork
    dfnew[1] = floors
    dfnew[2] = rooms
    dfnew[3] = area

    dfinputcol = list(dfinputnew.columns.values)
    for x in dfinputcol[4:30]:
        if district == x:
            dfnew[dfinputcol.index(x)] = True
        else:
            dfnew[dfinputcol.index(x)] = False
    
    for x in dfinputcol[30:-4]:
        if ward == x:
            dfnew[dfinputcol.index(x)] = True
        else:
            dfnew[dfinputcol.index(x)] = False

    for x in dfinputcol[-4:]:
        if houseType == x:
            dfnew[dfinputcol.index(x)] = True
        else:
            dfnew[dfinputcol.index(x)] = False

    #dfinputnew = dfinputnew.append(dfnew)
    input_dataframe = pd.DataFrame([dfnew], columns=dfinputcol)
    input_dataframe.to_csv('test_output.csv')
    to_be_scaled = ["Floors","Number of rooms","Area(m2)"]
    input_dataframe[to_be_scaled] = scaler.transform(input_dataframe[to_be_scaled])
    
    predict = model.predict(input_dataframe)[0]
    return render_template('index.html', predict=round(predict, 2))
    

if __name__ == "__main__":
    app.run(debug=True)

    