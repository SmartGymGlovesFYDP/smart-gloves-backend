import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.signal import butter,filtfilt
from scipy.spatial.distance import euclidean

# Import libraries and classes required for this example:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

BENCH = "./Data/Bench/"
CURLS = "./Data/Curls/"
TRICEPS = "./Data/Triceps/"

mapping = {
    0: "Bench Press",
    1: "Bicep Curls",
    2: "Tricep Extension"
}

injury = { # Compare the Right + Left Glove
    "Ax_Right >> Ax_Left" : "Right hand motion too fast (X-Direction)",
    "Ay_Right >> Ay_Left" : "Right hand motion too fast (Y-Direction)",
    "Az_Right >> Az_Left" : "Right hand motion too fast (Z-Direction)",
    "Gx_Right >> Gx_Left" : "Possible injury risk of rotational motion (X-Direction)",
    "Gy_Right >> Gy_Left" : "Possible injury risk of rotational motion (Y-Direction)",
    "Gz_Right >> Gz_Left" : "Possible injury risk of rotational motion (Z-Direction)",
    "Ax_Right << Ax_Left" : "Left hand motion too fast (X-Direction)",
    "Ay_Right << Ay_Left" : "Left hand motion too fast (Y-Direction)",
    "Az_Right << Az_Left" : "Left hand motion too fast (Z-Direction)",
    "Gx_Right << Gx_Left" : "Possible injury risk of rotational motion (X-Direction)",
    "Gy_Right << Gy_Left" : "Possible injury risk of rotational motion (Y-Direction)",
    "Gz_Right << Gz_Left" : "Possible injury risk of rotational motion (Z-Direction)"
    "Ax_Right ~ Ax_Left" : "Great speed for both hands (X-Direction)",
    "Ay_Right ~ Ay_Left" : "Great speed for both hands (Y-Direction)",
    "Az_Right ~ Az_Left" : "Great speed for both hands (Z-Direction)",
    "Gx_Right ~ Gx_Left" : "Great stability of rotational motion for both hands (X-Direction)",
    "Gy_Right ~ Gy_Left" : "Great stability of rotational motion for both hands (Y-Direction)",
    "Gz_Right ~ Gz_Left" : "Great stability of rotational motion for both hands (Z-Direction)"
}

properAvgBenchPressMSE = [0.011716247,	0.036809727,	0.072262675,    # Ax_Right, Ay_Right, Az_Right
                        1366.580343,	132.7365875,	192.725278,     # Gx_Right, Gy_Right, Gz_Right
                        0.013828273,	0.036533363,	0.039886446,    # Ax_Left, Ay_Left, Az_Left
                        2049.702587,	162.7141079,	716.8502251]    # Gx_Left, Gy_Left, Gz_Left

properAvgBicepCurlsMSE = [0.071671827,	0.290808289,	0.196147715,    # Ax_Right, Ay_Right, Az_Right
                        3414.610078,	3322.766288,	2681.099395,    # Gx_Right, Gy_Right, Gz_Right
                        0.087521492,	0.244233338,	0.195569624,    # Ax_Left, Ay_Left, Az_Left
                        1576.773251,	4018.62555,	    2932.637815]    # Gx_Left, Gy_Left, Gz_Left

properAvgTricepsMSE =   [0.13109381,	0.081085661,	0.0358325,      # Ax_Right, Ay_Right, Az_Right
                        659.4008787,	544.9047889,	1557.480128,    # Gx_Right, Gy_Right, Gz_Right
                        0.051912554,	0.056446856,	0.024493676,    # Ax_Left, Ay_Left, Az_Left
                        200.6815757,	226.2440168,	924.9710044]    # Gx_Left, Gy_Left, Gz_Left

def readJSON(filename):
    f = open(filename)
    data = json.load(f)
    f.close()
    return data

def extractData(sampleData):
    # To store timestamp values
    timeAxis = np.array([])

    # To store right glove data
    axRightAxis = np.array([])
    ayRightAxis = np.array([])
    azRightAxis = np.array([])
    gxRightAxis = np.array([])
    gyRightAxis = np.array([])
    gzRightAxis = np.array([])

    # To store left glove data
    axLeftAxis = np.array([])
    ayLeftAxis = np.array([])
    azLeftAxis = np.array([])
    gxLeftAxis = np.array([])
    gyLeftAxis = np.array([])
    gzLeftAxis = np.array([])

    # To append timestamp values
    for time in sampleData:
        # Get length of packets in single timestamp
        uidList = list(sampleData[time].keys())
        # Filter the objects that have both glove data values
        if (len(uidList) == 2):
            # Get the list of UID
            uidRight = list(sampleData[time]['rightGlove'].keys())
            uidLeft = list(sampleData[time]['leftGlove'].keys())
            tempTime = float(time.replace(":",""))
            timeAxis = np.append(timeAxis, tempTime)
            # Append to respective list (Right Glove)
            axRightAxis = np.append(axRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['ax']))
            ayRightAxis = np.append(ayRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['ay']))
            azRightAxis = np.append(azRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['az']))
            gxRightAxis = np.append(gxRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['gx']))
            gyRightAxis = np.append(gyRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['gy']))
            gzRightAxis = np.append(gzRightAxis, float(sampleData[time]['rightGlove'][uidRight[0]]['gz']))
            # Append to respective list (Left Glove)
            axLeftAxis = np.append(axLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['ax']))
            ayLeftAxis = np.append(ayLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['ay']))
            azLeftAxis = np.append(azLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['az']))
            gxLeftAxis = np.append(gxLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['gx']))
            gyLeftAxis = np.append(gyLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['gy']))
            gzLeftAxis = np.append(gzLeftAxis, float(sampleData[time]['leftGlove'][uidLeft[0]]['gz']))
    
    return timeAxis, axRightAxis, ayRightAxis, azRightAxis, gxRightAxis, gyRightAxis, gzRightAxis, axLeftAxis, ayLeftAxis, azLeftAxis, gxLeftAxis, gyLeftAxis, gzLeftAxis

def getLinearPoint(m, b, x):
    y = m*x + b
    return y

def percentageDifference(val1, val2):
    numerator = abs(val1-val2)
    denominator = abs((val1+val2)/2)
    return (numerator/denominator)*100

def calculateMSE(xDataAxis, yDataAxis, m, b):
    # n = total number of terms for which the error is calculated
    # y = the observed value
    # y_bar = predicted value (point on the regression line)
    y = yDataAxis
    n = len(xDataAxis)
    summation = 0
    # Loop through all available timestamp points
    for i in range(0, n):
        # This is the predicted value (derrived from the slope)
        y_bar = getLinearPoint(m, b, xDataAxis[i])
        # print(f"y_bar = {y_bar}")
        difference = y[i]-y_bar
        squared_difference = difference**2
        summation = summation + squared_difference
    MSE = summation/n
    return MSE

def plotData(xAxis, yAxis, zAxis, timeAxis, xLabel, yLabel, zLabel, filename):
    plt.minorticks_off()
    plt.scatter(timeAxis, xAxis, label=xLabel)
    plt.scatter(timeAxis, yAxis, label=yLabel)
    plt.scatter(timeAxis, zAxis, label=zLabel)

    m1, b1 = np.polyfit(timeAxis, xAxis, 1)
    m2, b2 = np.polyfit(timeAxis, yAxis, 1)
    m3, b3 = np.polyfit(timeAxis, zAxis, 1)

    # print (f"m1 = {m1}")
    # print (f"b1 = {b1}")

    mse1 = calculateMSE(timeAxis, xAxis, m1, b1)
    mse2 = calculateMSE(timeAxis, yAxis, m2, b2)
    mse3 = calculateMSE(timeAxis, zAxis, m3, b3)

    # print(f"MSE (X-Axis) = {mse1}")
    # print(f"MSE (Y-Axis) = {mse2}")
    # print(f"MSE (Z-Axis) = {mse3}")

    # plt.plot(timeAxis, m1*timeAxis + b1)
    # plt.plot(timeAxis, m2*timeAxis + b2)
    # plt.plot(timeAxis, m3*timeAxis + b3)

    # plt.legend()
    # plt.show()
    # plt.savefig(filename)
    # plt.clf()

    # Return the MSE of X Axis, Y Axis, Z Axis
    return mse1, mse2, mse3

def generateSlopes(workoutData, workoutName):
    # To hold all the slopes for a certain exercise including the perfect model
    tempSlopes = []

    fileCount = 0
    for data in workoutData:
        slopes = [0]*12
        fileCount += 1
        fileNameRightA = "" + workoutName + "_A_Right_" + str(fileCount) + ".png"
        fileNameRightG = "" + workoutName + "_G_Right_" + str(fileCount) + ".png"
        fileNameLeftA = "" + workoutName + "_A_Left_" + str(fileCount) + ".png"
        fileNameLeftG = "" + workoutName + "_G_Left_" + str(fileCount) + ".png"
        timeAxis, axRightAxis, ayRightAxis, azRightAxis, gxRightAxis, gyRightAxis, gzRightAxis, axLeftAxis, ayLeftAxis, azLeftAxis, gxLeftAxis, gyLeftAxis, gzLeftAxis = extractData(data)
  
        slopes[0], slopes[1], slopes[2] = plotData(axRightAxis, ayRightAxis, azRightAxis, timeAxis, "Right (Ax)", "Right (Ay)", "Right (Az)", fileNameRightA)
        slopes[3], slopes[4], slopes[5] = plotData(gxRightAxis, gyRightAxis, gzRightAxis, timeAxis, "Right (Gx)", "Right (Gy)", "Right (Gz)", fileNameRightG)

        slopes[6], slopes[7], slopes[8] = plotData(axLeftAxis, ayLeftAxis, azLeftAxis, timeAxis, "Left (Ax)", "Left (Ay)", "Left (Az)", fileNameLeftA)
        slopes[9], slopes[10], slopes[11] = plotData(gxLeftAxis, gyLeftAxis, gzLeftAxis, timeAxis, "Left (Gx)", "Left (Gy)", "Left (Gz)", fileNameLeftG)

        tempSlopes.append(slopes)
    
    return tempSlopes

######      THIS IS TO GIVE A RATING OF THE WORKOUT
# [DONE] 1) User trains with proper form (used 4 reference points for each workout)
# [DONE] 2.1) Using linear regression, a slope formed for the trained data => Fix this to a slope + y-intercept to compare it later
# [DONE] 2.2) Mean Squared Error is calculated from the slope and the actual data => Fix this to a number to compare it later
# [DONE] 2.3) Averaged the MSE from the workout of proper form to be used for comparison
# [DONE] 3.1) With trial data, use linear regression to get a slope for the trial data
# [DONE] 3.2) Calculate Mean Squared Error from the slope and scattered points
# [] 3.3) Find the differences in each direction and give a rating and injury risk in each direction
# [] 4) Compare the Mean Squared Error (Expected vs Actual) find the percentage difference and evaluate the workout
def rateWorkout(allSlopes, workout):
    diffInWorkout = []
    rating = 0

    if workout == "Bench Press":
        for i in range(len(allSlopes)):
            diffInWorkout.append(percentageDifference(allSlopes[i], properAvgBenchPressMSE[i]))
    elif workout == "Bicep Curls":
        for i in range(len(allSlopes)):
            diffInWorkout.append(percentageDifference(allSlopes[i], properAvgBicepCurlsMSE[i]))
    elif workout == "Triceps Extension":
        for i in range(len(allSlopes)):
            diffInWorkout.append(percentageDifference(allSlopes[i], properAvgTricepsMSE[i]))
    else:
        return "Invalid Input!"

    return diffInWorkout

def predictWorkout(slopes_X_test):
    # Import dataset:
    url = "All.csv"

    # Assign column names to dataset:
    names = ['ax_Right','ay_Right','az_Right','gx_Right','gy_Right','gz_Right','ax_Left','ay_Left','az_Left','gx_Left','gy_Left','gz_Left','exercise']
    
    # Convert dataset to a pandas dataframe:
    dataset = pd.read_csv(url, names=names) 

    # Use head() function to return the first 5 rows: 
    dataset.head()

    # Assign values to the X and y variables:
    X_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, 12].values 

    # Serves as a predictor (0 = BP, 1 = BC, 2 = TE)
    slopes_y_test = -1

    # Standardize features by removing mean and scaling to unit variance:
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    slopes_X_test = scaler.transform(slopes_X_test) 

    # Use the KNN classifier to fit data:
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train) 

    # Predict y data with classifier: 
    slopes_y_test = classifier.predict(slopes_X_test)
    # print(mapping[int(slopes_y_test)])

    return int(slopes_y_test)
        
def main():

    # # To hold all the slopes for a certain exercise
    # allSlopes = []
    
    # workoutData = []
    # workout = readJSON("workout.json")
    # workoutData.append(workout)

    # allSlopes = generateSlopes(workoutData, "Workout")
    # Sample test slopes to see what it returns
    # allSlopes = [[0.071438754,0.289096595,0.210599216,2384.568664,2488.04925,1789.299704,0.097597208,0.295178053,0.200608988,1333.62713,2985.354642,2148.434388]]
    # print (allSlopes)
    # 0 = Bench Press   = [[0.022603186,0.00243637,0.02918603,5.339126702,107.8320342,7.000924645,0.028759372,0.008630163,0.02132904,22.92064783,99.48017482,3.45258442]]
    # 1 = Bicep Curls   = [[0.071438754,0.289096595,0.210599216,2384.568664,2488.04925,1789.299704,0.097597208,0.295178053,0.200608988,1333.62713,2985.354642,2148.434388]]
    # 2 = Triceps       = [[0.161019347,0.106992285,0.047709297,442.3707667,724.1853624,1984.234125,0.033414772,0.076311812,0.028354947,221.5938891,245.4956944,1074.739071]]
    # guessWorkout = predictWorkout(allSlopes)
    # print(mapping[guessWorkout])

    # workoutRating = rateWorkout(allSlopes[0], guessWorkout)
    # print(workoutRating)

    # myDf = pd.DataFrame(workoutData)
    # myDf.to_csv('output_workout.csv', index=False, header=False)

    # To hold all the slopes for a certain exercise including the perfect model
    allBenchPressSlopes = []
    allCurlsSlopes = []
    allTricepSlopes = []
    
    benchPressData = []
    benchPressPerfect1 = readJSON(BENCH + "bench0.json")
    benchPressPerfect2 = readJSON(BENCH + "bench1.json")
    benchPressPerfect3 = readJSON(BENCH + "bench2.json")
    benchPressData.append(benchPressPerfect1)
    benchPressData.append(benchPressPerfect2)
    benchPressData.append(benchPressPerfect3)

    curlsData = []
    curlsPerfect1 = readJSON(CURLS + "curls0.json")
    curlsPerfect2 = readJSON(CURLS + "curls1.json")
    curlsPerfect3 = readJSON(CURLS + "curls2.json")
    curlsData.append(curlsPerfect1)
    curlsData.append(curlsPerfect2)
    curlsData.append(curlsPerfect3)

    tricepsData = []
    tricepsPerfect1 = readJSON(Triceps + "triceps0.json")
    tricepsPerfect2 = readJSON(Triceps + "triceps1.json")
    tricepsPerfect3 = readJSON(Triceps + "triceps2.json")
    tricepsData.append(tricepsPerfect1)
    tricepsData.append(tricepsPerfect2)
    tricepsData.append(tricepsPerfect3)

    allBenchPressSlopes = generateSlopes(benchPressData, "BenchPress")
    allCurlsSlopes = generateSlopes(curlsData, "Curls")
    allTricepSlopes = generateSlopes(tricepsData, "Triceps")

    myDf1 = pd.DataFrame(allBenchPressSlopes)
    myDf2 = pd.DataFrame(allCurlsSlopes)
    myDf3 = pd.DataFrame(allTricepSlopes)

    myDf1.to_csv('output_bench_press.csv', index=False, header=False)
    myDf2.to_csv('output_curls.csv', index=False, header=False)
    myDf3.to_csv('output_triceps.csv', index=False, header=False)

if __name__ == '__main__':
    main()

'''
TO-DOs:

Functional

[SHAHIL]
1) Display analyzed smart gym glove sensor data for at least 3 distinct gym activities with 2 or more metrics
* [Back-End] 

    - Bench Press
        Ax_Right = 
        Ay_Right = 
        Az_Right = 
        Gx_Right = 
        Gy_Right = 
        Gz_Right = 

        Ax_Left = 
        Ay_Left = 
        Az_Left = 
        Gx_Left = 
        Gy_Left = 
        Gz_Left = 

    - Bicep Curl
        Ax_Right = 
        Ay_Right = 
        Az_Right = 
        Gx_Right = 
        Gy_Right = 
        Gz_Right = 

        Ax_Left = 
        Ay_Left = 
        Az_Left = 
        Gx_Left = 
        Gy_Left = 
        Gz_Left = 

    - Tricep Overhead Extension
        Ax_Right = 
        Ay_Right = 
        Az_Right = 
        Gx_Right = 
        Gy_Right = 
        Gz_Right = 

        Ax_Left = 
        Ay_Left = 
        Az_Left = 
        Gx_Left = 
        Gy_Left = 
        Gz_Left = 

* [Front-End] Being able to extract workout history from the backend and show the results in the frontend.
* [Test] Are we able to view the results in the mobile app?

2) The app must have a library of 20 distinct exercises with the ability for the user to add at least 50 custom exercises.
* [Front-End] Add a form to the front-end to add a new exercise to the library
* [Test] Are we able to add a new exercise and does it show up in the backend?

3) A user can view workout analysis and results for day, week, month, and year with scores based on 2 or more key metrics.
* [Back-End] Nothing really, the two metrics will be generated from the available workout history from firestore.
* [Front-End] Will have to filter and provide the score on Arms/Chest/Legs for day/week/month/year. 
* [Test] Are we able to view this analysis in the mobile app?

4) The app must have 5 methods to gamify workouts and compete with friends. This includes workout streaks, badges, trophies, leaderboards, and scores.
* [Back-End] Will have to create a friends list collection and connect their workout history performance.
* [Front-End] Being able to view the performances/metrics of friends on the app.
* [Test] Are the results shown on the mobile app?

5) A user can sign into any mobile phone with the Smart Gym app and load the home screen within 10 seconds.
* [Test] Compute the timing from signing into the mobile app?

[SHAHIL]
6) Automatically recognize 3 distinct gym activities using smart gym glove data with 95% accuracy.
* [Back-End] Using
        Input = Raw Data
        Process = Raw Data -> Convert to Slopes (12 slopes) -> Use that to train KNN Model -> Test the Raw Data Slopes with Model -> Return Exercise
        Output = Classify the workout and place it properly in the workoutHistory
* [Test] Test 10 workouts, it should yield 10/10 to achieve over 95% accuracy

7) The backend server must convert sensor data into readable metrics within 30 seconds of a user completing an activity.
* [Test] Compute the timing from finishing workout to viewing the results in the mobile app?


Non-Functional

1) The mobile app must be supported for Android 5.0+ and iOS 13+ compatible devices.
* [Test] Refer to the EXPO guidelines? What versions can we export the Expo Project to in Android and Apple?

2) Incorporate 5 mobile native health and fitness metrics to improve ML based injury risk assessment.
* [Back-End] N/A
* [Front-End] N/A
* [Test] N/A

3) The backend server must have an uptime of 95%.
* [Test] Are we able to compute the results from workout and view it in mobile app? If yes, then we know backend is up.

4) The delay from API calls between the database and the mobile application must be within 750 milliseconds.
* [Test] Compute the timing of the calls from database and mobile app?

'''