import os
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app, db
from enum import Enum
import datetime

app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
firestore_db = firestore.client()
users_ref = firestore_db.collection('users')
workouts_ref = firestore_db.collection('workouts')
realtimedata_ref = db.reference(
    '/', None, 'https://smartgloves-e450e-default-rtdb.firebaseio.com/')

# Specific Document + Collection to extract the user's workout
specificUser = 'XJGnJ3aJ3zUWcUAgtQOsie8oM9Y2'
newWorkout_ref = users_ref.document(specificUser).collection('newWorkout')
workoutHistory_ref = users_ref.document(
    specificUser).collection('workoutHistory')

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
            dt = datetime.datetime(2022, 2, 24, 8, 00, 00)
            workouts = workoutHistory_ref.where('timestamp', ">=", dt).stream()
            return workouts[0]['intensity']
        else:  # POST
            if(heart_rate > -1):
                return calcIntensityHeartRate(heart_rate)
            else:
                # TODO: update this to use the post request values as parameters
                return calcIntensity()
    except Exception as e:
        return f"An error occurred when determining intensity: {e}"


@app.route("/detectWorkout", methods=['GET'])
def detectWorkout():
    try:
        newWorkout = [doc.to_dict() for doc in newWorkout_ref.stream()]
        # Ideally, we will only have 1 temporary workout that will be processed so grab the first one
        workoutName = newWorkout[0]['workoutName']
        workoutMajorMuscle = newWorkout[0]['majorMuscle']
        workoutDifficulty = newWorkout[0]['difficulty']
        workoutMinutes = newWorkout[0]['minutes']
        return workoutName, workoutMajorMuscle, workoutDifficulty, workoutMinutes, 200
    except Exception as e:
        return f"An error occurred when determining temp workout: {e}"


@app.route("/updateWorkout", methods=['GET'])
def updateWorkout():
    try:
        tempWorkoutInfo = detectWorkout()
        if tempWorkoutInfo[0] != "A":
            # update the workout history
            workoutHistory_ref.add({
                "workoutName": tempWorkoutInfo[0],
                "duration": tempWorkoutInfo[3],
                "intensity": calcIntensity(tempWorkoutInfo[2], tempWorkoutInfo[3]),
                "performance": calcPerformance(tempWorkoutInfo[3]),
                "majorMuscle": tempWorkoutInfo[1],
                "timestamp": datetime.datetime.now()
            })
            # clear the documents in the newWorkout_ref so only a single document exists at a time
            for doc in newWorkout_ref.stream():
                doc.reference.delete()
            return "done", 200
        else:
            return "no changes made", 200
    except Exception as e:
        return f"An error occurred when updateing workout summary: {e}"


# Define Middleware below


class Intensity(Enum):
    LOW = 0.1
    MID = 0.2
    HIGH = 0.3
    MAXED = 0.4


def calcIntensity(workoutDifficulty, workoutMinutes):
    try:
        intensity = workoutDifficulty / 5
        if workoutMinutes <= 10:
            intensity += Intensity.LOW
        elif workoutMinutes <= 45:
            intensity += Intensity.MID
        else:
            intensity += Intensity.HIGH
        return intensity
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


def getGloveData():
    # one time call to get current instance of the realtime db
    data = realtimedata_ref.get()
    # stream to catch live data
    dataStream = realtimedata_ref.listen(listenHandler)


def listenHandler(event):
    # example of value access: event.data['ax']
    print("data", event.data)


def calcPerformance(workoutMinutes):
    # TODO: implimented soon, should be a percentage
    if workoutMinutes <= 45:
        return 0.96
    else:
        return 0.72


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port='8000')
