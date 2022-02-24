import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.signal import butter,filtfilt
from scipy.spatial.distance import euclidean

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

    plt.plot(timeAxis, m1*timeAxis + b1)
    plt.plot(timeAxis, m2*timeAxis + b2)
    plt.plot(timeAxis, m3*timeAxis + b3)

    plt.legend()
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


######      THIS IS TO PREDICT THE WORKOUT
# 1) Compare and add the absolute value of the differences in points of each workout
# 2) Add the differences
# 3) Return the workout with the lowest difference
def predictWorkout(allSlopes):
    # Variables to hold the workout difference value compared to the proper form
    diffBP = 0.0
    diffC = 0.0
    diffT = 0.0

    for i in range (len(allSlopes)):
        diffBP += abs(allSlopes[i] - properAvgBenchPressMSE[i])
        diffC += abs(allSlopes[i] - properAvgBicepCurlsMSE[i])
        diffT += abs(allSlopes[i] - properAvgTricepsMSE[i])

    print (diffBP)
    print (diffC)
    print (diffT)

    if diffBP < diffC and diffBP < diffT:
        return "Bench Press"
    elif diffC < diffBP and diffC < diffT:
        return "Bicep Curls"
    elif diffT < diffBP and diffT < diffC:
        return "Triceps Extension"
    else:
        return "Failure to recognize workout"
        
def main():

    # To hold all the slopes for a certain exercise
    allSlopes = []
    
    workoutData = []
    workout = readJSON("workout6.json")
    workoutData.append(workout)

    allSlopes = generateSlopes(workoutData, "Workout")
    guessWorkout = predictWorkout(allSlopes[0])

    print (allSlopes)
    print(guessWorkout)

    workoutRating = rateWorkout(allSlopes[0], guessWorkout)
    # print(workoutRating)

    # myDf = pd.DataFrame(workoutData)
    # myDf.to_csv('output_workout.csv', index=False, header=False)

if __name__ == '__main__':
    main()