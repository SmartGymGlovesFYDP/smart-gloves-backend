import os
from flask import Flask, request
from firebase_admin import credentials, firestore, initialize_app
from enum import Enum

app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
users_ref = db.collection('users')
workouts_ref = db.collection('workouts')


# Define Routes below


@app.route("/")
def init():
    return "pass in a valid route"


@app.route("/exerciseData")
def exerciseData():
    return {"intensity": ["Mad", "Bad", "Sad"]}


@app.route("/intensity", defaults={'heart_rate': -1}, methods=['GET', 'POST'])
@app.route("/intensity/<heart_rate>", methods=['GET', 'POST'])
def intensity(heart_rate):
    try:
        if request.method == 'GET':
            return showIntensity()
        else:  # POST
            if(heart_rate > -1):
                return calcIntensityHeartRate(heart_rate)
            else:
                return calcIntensity()
    except Exception as e:
        return f"An error occurred when determining intensity: {e}"


# Define Middleware below
class Intensity(Enum):
    LOW = 1
    MID = 2
    HIGH = 3
    MAXED = 4


def showIntensity():
    # return should be a read from firebase user data on past workout information
    # remember the most recent past is the current workout
    return Intensity.LOW.value


def calcIntensity():  # TODO: after moving some mock user data into firebase we can finish this
    # default
    # step 1) make calculation
    # step 2) save to firebase
    # step 3) return calc
    try:
        user_id = request.args.get('id')
        if user_id:
            user_Session_Data = users_ref.document(user_id).get()
            # TODO: search for the given data about the session
            # TODO: factors to account for intensity
            # 1) variation difficulty, majorMuscle in order of highest intensity ('Full Body', 'Legs', 'Arms','Core'), number of majorMuscle groups targeted, number of minorMuscle groups targeted
        return Intensity.MAXED.value
    except Exception as e:
        return f"An error occurred while calculating intensity: {e}"


# TODO: modification needed based on how heartRate data will be sent
def calcIntensityHeartRate(heartRate):
    # calculation with heart rate
    userAge = 20  # TODO: pull value from Firebase
    userRestingHeartRate = 65  # TODO: pull value from Firebase
    maxHeartRate = 220 - userAge
    heartRateReserve = maxHeartRate - userRestingHeartRate
    userIntensityArray = []
    if(heartRate < heartRateReserve * 0.65):
        userIntensityArray.append(Intensity.LOW)
    elif(heartRate < heartRateReserve * 0.7 and heartRate >= heartRateReserve * 0.65):
        userIntensityArray.append(Intensity.MID)
    elif(heartRate <= heartRateReserve * 0.85 and heartRate >= heartRateReserve * 0.7):
        userIntensityArray.append(Intensity.HIGH)
    else:
        userIntensityArray.append(Intensity.MAXED)
    return {"userIntensity": userIntensityArray}


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port='8000')
