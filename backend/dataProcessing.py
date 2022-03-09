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

BENCH = "./Data_Processing_Algorithm/Data/Bench/"
CURLS = "./Data_Processing_Algorithm/Data/Curls/"
TRICEPS = "./Data_Processing_Algorithm/Data/Triceps/"
TESTML = "./TestML/"

mapping = {
    0: "Bench Press",
    1: "Bicep Curls",
    2: "Tricep Extension"
}

injury = { # Compare the Raw Data Slopes to the Error Bar of LOW-HIGH values
    0 : "hand motion too fast in the X-Direction",                          # Ax_Raw > Ax_Proper
    1 : "hand motion too slow in the X-Direction",                          # Ax_Raw < Ax_Proper
    2 : "hand motion speed is perfect in the X-Direction",                  # Ax_Raw ~ Ax_Proper 
    3 : "hand motion too fast in the Y-Direction",                          # Ay_Raw > Ay_Proper
    4 : "hand motion too slow in the Y-Direction",                          # Ay_Raw < Ay_Proper
    5 : "hand motion speed is perfect in the Y-Direction",                  # Ay_Raw ~ Ay_Proper
    6 : "hand motion too fast in the Z-Direction",                          # Az_Raw > Az_Proper
    7 : "hand motion too slow in the Z-Direction",                          # Az_Raw < Az_Proper
    8 : "hand motion speed is perfect in the Z-Direction",                  # Az_Raw ~ Az_Proper
    9 : "possible injury risk of rotational motion in the X-Direction",     # Gx_Raw >|< Gx_Proper
    10: "great rotational motion stability in the X-Direction",             # Gx_Raw ~ Gx_Proper
    11: "possible injury risk of rotational motion in the Y-Direction",     # Gy_Raw >|< Gy_Proper
    12: "great rotational motion stability in the Y-Direction",             # Gy_Raw ~ Gy_Proper
    13: "possible injury risk of rotational motion in the Z-Direction",     # Gz_Raw >|< Gz_Proper
    14: "great rotational motion stability in the Z-Direction"              # Gz_Raw ~ Gz_Proper
}

properAvgBenchPress =    [(0.000954270, 0.004019794), # (Ax_Right_LOW,Ax_Right_HIGH)
                          (0.019308498, 0.029403562), # (Ay_Right_LOW,Ay_Right_HIGH)
                          (0.014885754, 0.036409837), # (Az_Right_LOW,Az_Right_HIGH)
                          (99.38704950, 330.4275585), # (Gx_Right_LOW,Gx_Right_HIGH)
                          (4.121318247, 12.60476745), # (Gy_Right_LOW,Gy_Right_HIGH)
                          (10.35889395, 22.21108908), # (Gz_Right_LOW,Gz_Right_HIGH)
                          (0.001629672, 0.003681337), # (Ax_Left_LOW,Ax_Left_HIGH)
                          (0.018381728, 0.028166604), # (Ay_Left_LOW,Ay_Left_HIGH)
                          (0.026646431, 0.042366096), # (Az_Left_LOW,Az_Left_HIGH)
                          (116.4247039, 306.5351855), # (Gx_Left_LOW,Gx_Left_HIGH)
                          (11.44071676, 37.58922647), # (Gy_Left_LOW,Gy_Left_HIGH)
                          (10.40063429, 25.69945336)] # (Gz_Left_LOW,Gz_Left_HIGH)

properAvgBicepCurls =    [(0.073003751, 0.156914016), # (Ax_Right_LOW,Ax_Right_HIGH)
                          (0.113470746, 0.378368239), # (Ay_Right_LOW,Ay_Right_HIGH)
                          (0.204006200, 0.338959347), # (Az_Right_LOW,Az_Right_HIGH)
                          (1490.716510, 2796.339401), # (Gx_Right_LOW,Gx_Right_HIGH)
                          (4231.833769, 5192.279753), # (Gy_Right_LOW,Gy_Right_HIGH)
                          (1483.068245, 2388.213538), # (Gz_Right_LOW,Gz_Right_HIGH)
                          (0.055049585, 0.095754317), # (Ax_Left_LOW,Ax_Left_HIGH)
                          (0.154317325, 0.364256575), # (Ay_Left_LOW,Ay_Left_HIGH)
                          (0.168226103, 0.290224076), # (Az_Left_LOW,Az_Left_HIGH)
                          (959.2773030, 3405.309227), # (Gx_Left_LOW,Gx_Left_HIGH)
                          (2821.313807, 4507.885355), # (Gy_Left_LOW,Gy_Left_HIGH)
                          (1062.203702, 2892.248853)] # (Gz_Left_LOW,Gz_Left_HIGH)

properAvgTriceps =       [(0.003766914, 0.057215426), # (Ax_Right_LOW,Ax_Right_HIGH)
                          (0.024402355, 0.203202160), # (Ay_Right_LOW,Ay_Right_HIGH)
                          (0.002333381, 0.013026269), # (Az_Right_LOW,Az_Right_HIGH)
                          (99.41960390, 330.3155266), # (Gx_Right_LOW,Gx_Right_HIGH)
                          (42.81418196, 123.1278526), # (Gy_Right_LOW,Gy_Right_HIGH)
                          (212.4638896, 1320.382979), # (Gz_Right_LOW,Gz_Right_HIGH)
                          (0.001674259, 0.034351184), # (Ax_Left_LOW,Ax_Left_HIGH)
                          (0.027319996, 0.219775514), # (Ay_Left_LOW,Ay_Left_HIGH)
                          (0.001772729, 0.020093009), # (Az_Left_LOW,Az_Left_HIGH)
                          (80.52329157, 314.9831677), # (Gx_Left_LOW,Gx_Left_HIGH)
                          (42.54601338, 87.95397931), # (Gy_Left_LOW,Gy_Left_HIGH)
                          (372.0777554, 1260.801085)] # (Gz_Left_LOW,Gz_Left_HIGH)

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
    # plt.minorticks_off()
    # plt.scatter(timeAxis, xAxis, label=xLabel)
    # plt.scatter(timeAxis, yAxis, label=yLabel)
    # plt.scatter(timeAxis, zAxis, label=zLabel)

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

def evaluateForm(raw, proper):
    overallScore, rightHandScore, leftHandScore, stars = 0, 0, 0, 0
    score = []
    tips = []

    for i in range(len(raw)):
        if i == 0: # RIGHT_AX
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Right " + injury[0])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Right " + injury[1])
            else:
                score.append(2)
                # tips.append("Right " + injury[2])
        elif i == 1: # RIGHT_AY
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Right " + injury[3])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Right " + injury[4])
            else:
                score.append(2)
                # tips.append("Right " + injury[5])
        elif i == 2: # RIGHT_AZ
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Right " + injury[6])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Right " + injury[7])
            else:
                score.append(2)
                # tips.append("Right " + injury[8])
        elif i == 3: # RIGHT_GX
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Right hand has " + injury[9])
            else:
                score.append(2)
                # tips.append("Right hand has " + injury[10])
        elif i == 4: # RIGHT_GY
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Right hand has " + injury[11])
            else:
                score.append(2)
                # tips.append("Right hand has " + injury[12])
        elif i == 5: # RIGHT_GZ
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Right hand has " + injury[13])
            else:
                score.append(2)
                # tips.append("Right hand has " + injury[14])
        elif i == 6: # LEFT_AX
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Left " + injury[0])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Left " + injury[1])
            else:
                score.append(2)
                # tips.append("Left " + injury[2])
        elif i == 7: # LEFT_AY
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Left " + injury[3])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Left " + injury[4])
            else:
                score.append(2)
                # tips.append("Left " + injury[5])
        elif i == 8: # LEFT_AZ
            if raw[i] > proper[i][1]: # RAW Greater than PROPER HIGH
                score.append(0)
                tips.append("Left " + injury[6])
            elif raw[i] < proper[i][0]: # RAW Less than PROPER LOW
                score.append(1)
                tips.append("Left " + injury[7])
            else:
                score.append(2)
                # tips.append("Left " + injury[8])
        elif i == 9: # LEFT_GX
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Left hand has " + injury[9])
            else:
                score.append(2)
                # tips.append("Left hand has " + injury[10])
        elif i == 10: # LEFT_GY
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Left hand has " + injury[11])
            else:
                score.append(2)
                # tips.append("Left hand has " + injury[12])
        elif i == 11: # LEFT_GZ
            if raw[i] > proper[i][1] or raw[i] < proper[i][0] : # RAW Greater/Less than PROPER HIGH
                score.append(0)
                tips.append("Left hand has " + injury[13])
            else:
                score.append(2)
                # tips.append("Left hand has " + injury[14])

    for i in range(len(raw)):
        overallScore += score[i]
        if i <= 5:
            rightHandScore += score[i]
        else:
            leftHandScore += score[i]

    overallScore = (overallScore/24)*100     # Metric 1 = Overall Workout + Stars
    rightHandScore = (rightHandScore/12)*100 # Metric 2 = Wellness of Right Hand
    leftHandScore = (leftHandScore/12)*100   # Metric 3 = Wellness of Left Hand
    if overallScore >= 90:
        stars = 5.0
    elif overallScore >= 80:
        stars = 4.5
    elif overallScore >= 70:
        stars = 4.0
    elif overallScore >= 60:
        stars = 3.5
    elif overallScore >= 50:
        stars = 3.0
    elif overallScore >= 40:
        stars = 2.5
    elif overallScore >= 30:
        stars = 2.0
    elif overallScore >= 20:
        stars = 1.5
    elif overallScore >= 10:
        stars = 1.0
    else:
        stars = 0.0

    ratings = []
    ratings.append(overallScore)
    ratings.append(rightHandScore)
    ratings.append(leftHandScore)
    ratings.append(stars)

    return ratings, tips

######      THIS IS TO GIVE A RATING OF THE WORKOUT
# [DONE] 1) User trains with proper form (used 5 reference points for each workout)
# [DONE] 2.1) Using linear regression, a slope formed for the trained data => Fix this to a slope + y-intercept to compare it later
# [DONE] 2.2) Mean Squared Error is calculated from the slope and the actual data => Fix this to a number to compare it later
# [DONE] 2.3) Averaged the MSE from the workout of proper form to be used for comparison
# [DONE] 3.1) With trial data, use linear regression to get a slope for the trial data
# [DONE] 3.2) Calculate Mean Squared Error from the slope and scattered points
# [DONE] 3.3) Find the differences in each direction and give a rating and injury risk in each direction
def rateWorkout(allSlopes, workout):
    if workout == "Bench Press":
        ratings, tips = evaluateForm(allSlopes, properAvgBenchPress)
    elif workout == "Bicep Curls":
        ratings, tips = evaluateForm(allSlopes, properAvgBicepCurls)
    elif workout == "Triceps Extension":
        ratings, tips = evaluateForm(allSlopes, properAvgTriceps)
    else:
        return "Invalid Input!"
    return ratings, tips

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

def testML():
    # To hold all the slopes for a certain exercise including the perfect model
    allBenchPressSlopes = []
    allCurlsSlopes = []
    allTricepSlopes = []
    
    benchPressData = []
    benchPress1 = readJSON(TESTML + "bench0.json")
    benchPress2 = readJSON(TESTML + "bench1.json")
    benchPress3 = readJSON(TESTML + "bench2.json")
    benchPressData.append(benchPress1)
    benchPressData.append(benchPress2)
    benchPressData.append(benchPress3)

    curlsData = []
    curls1 = readJSON(TESTML + "curls0.json")
    curls2 = readJSON(TESTML + "curls1.json")
    curls3 = readJSON(TESTML + "curls2.json")
    curls4 = readJSON(TESTML + "curls3.json")
    curlsData.append(curls1)
    curlsData.append(curls2)
    curlsData.append(curls3)
    curlsData.append(curls4)

    tricepsData = []
    triceps1 = readJSON(TESTML + "triceps0.json")
    triceps2 = readJSON(TESTML + "triceps1.json")
    triceps3 = readJSON(TESTML + "triceps2.json")
    tricepsData.append(triceps1)
    tricepsData.append(triceps2)
    tricepsData.append(triceps3)

    allBenchPressSlopes = generateSlopes(benchPressData, "BenchPress")
    allCurlsSlopes = generateSlopes(curlsData, "Curls")
    allTricepSlopes = generateSlopes(tricepsData, "Triceps")

    print("TEST 1: Bench Press: ", "Bench Press" == mapping[predictWorkout([allBenchPressSlopes[0]])] )
    print("TEST 2: Bench Press: ", "Bench Press" == mapping[predictWorkout([allBenchPressSlopes[1]])] )
    print("TEST 3: Bench Press: ", "Bench Press" == mapping[predictWorkout([allBenchPressSlopes[2]])] )
    print("TEST 4: Bicep Curls: ", "Bicep Curls" == mapping[predictWorkout([allCurlsSlopes[0]])] )
    print("TEST 5: Bicep Curls: ", "Bicep Curls" == mapping[predictWorkout([allCurlsSlopes[1]])] )
    print("TEST 6: Bicep Curls: ", "Bicep Curls" == mapping[predictWorkout([allCurlsSlopes[2]])] )
    print("TEST 7: Bicep Curls: ", "Bicep Curls" == mapping[predictWorkout([allCurlsSlopes[3]])] ) 
    print("TEST 8: Tricep Extension: ", "Tricep Extension" == mapping[predictWorkout([allTricepSlopes[0]])] ) 
    print("TEST 9: Tricep Extension: ", "Tricep Extension" == mapping[predictWorkout([allTricepSlopes[1]])] ) 
    print("TEST 10: Tricep Extension: ", "Tricep Extension" == mapping[predictWorkout([allTricepSlopes[2]])] ) 
     
def dataProcess(workout):
    
    if workout:

        # testML()

        # To hold all the slopes for a certain exercise
        allSlopes = []

        workoutData = []
        # workout = getGloveData() # (UN)COMMENT THIS TO TEST FIREBASE REALTIMEDB
        # workout = readJSON("workout1.json") # (UN)COMMENT THIS TO TEST LOCAL JSON FILES
        workoutData.append(workout)

        allSlopes = generateSlopes(workoutData, "Workout")
        # Sample test slopes to see what it returns
        # allSlopes = [[0.071438754,0.289096595,0.210599216,2384.568664,2488.04925,1789.299704,0.097597208,0.295178053,0.200608988,1333.62713,2985.354642,2148.434388]]
        # print (allSlopes)
        # 0 = Bench Press   = [[0.022603186,0.00243637,0.02918603,5.339126702,107.8320342,7.000924645,0.028759372,0.008630163,0.02132904,22.92064783,99.48017482,3.45258442]]
        # 1 = Bicep Curls   = [[0.071438754,0.289096595,0.210599216,2384.568664,2488.04925,1789.299704,0.097597208,0.295178053,0.200608988,1333.62713,2985.354642,2148.434388]]
        # 2 = Triceps       = [[0.161019347,0.106992285,0.047709297,442.3707667,724.1853624,1984.234125,0.033414772,0.076311812,0.028354947,221.5938891,245.4956944,1074.739071]]
        guessWorkout = predictWorkout(allSlopes)
        predictedWorkout = mapping[guessWorkout]
        # print(predictedWorkout)

        workoutRating, tips = rateWorkout(allSlopes[0], predictedWorkout)
        # print(workoutRating)
        # print(tips)

        workoutResults = {
            "predictedWorkout": predictedWorkout,
            "overallScore": workoutRating[0],
            "rightHandScore": workoutRating[1],
            "leftHandScore": workoutRating[2],
            "stars": workoutRating[3],
            "tips": tips
        }

        return workoutResults

    return 0

'''
TO-DOs:

Functional

[SHAHIL]
1) Display analyzed smart gym glove sensor data for at least 3 distinct gym activities with 2 or more metrics
* [Back-End] Perform analysis on the workout
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