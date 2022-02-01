from crypt import methods
from flask import Flask,jsonify,request
from classifer import GetPrediction

app=Flask(__name__)

@app.route("/Predict-digit",methods=["POST"])

def PredictData():
    Image=request.files.get("digit")
    prediction=GetPrediction(Image)

    return jsonify({
        "prediction":prediction
    }),200

if __name__=="__main__":
    app.run(debug=True)