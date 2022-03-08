import numpy as np
from scipy.signal import butter,filtfilt

# FILTERs can be used if it improves the exercise recognition
# Filter requirements.
T = 3.0         # Sample Period
fs = 25.0       # sample rate, Hz
cutoff = 3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 4       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def plotSingleData(xAxis, yAxis, label, filename):
    plt.minorticks_off()
    plt.plot(yAxis, xAxis, label=label)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():

    ## EXAMPLE TO FILTER THE WORKOUT DATA
    # # Filter the data, and plot both the original and filtered signals.
    # y = butter_lowpass_filter(axRightAxis, cutoff, fs, order)

    # plt.subplot(2, 1, 2)
    # plt.plot(timeAxis, axRightAxis, 'b-', label='axRightAxis')
    # plt.plot(timeAxis, y, 'g-', linewidth=2, label='filtered axRightAxis')
    # plt.xlabel('Time [sec]')
    # plt.grid()
    # plt.legend()

    # plt.subplots_adjust(hspace=0.35)
    # plt.show()

    ## EXAMPLE TO PLOT SINGLE DATA
    # plotSingleData(axRightAxis, timeAxis, "Right (Ax)", "Ax_Right.png")
    # plotSingleData(ayRightAxis, timeAxis, "Right (Ay)", "Ay_Right.png")
    # plotSingleData(azRightAxis, timeAxis, "Right (Az)", "Az_Right.png")
    # plotSingleData(gxRightAxis, timeAxis, "Right (Gx)", "Gx_Right.png")
    # plotSingleData(gyRightAxis, timeAxis, "Right (Gy)", "Gy_Right.png")
    # plotSingleData(gzRightAxis, timeAxis, "Right (Gz)", "Gz_Right.png")
    
    # plotSingleData(axLeftAxis, timeAxis, "Left (Ax)", "Ax_Left.png")
    # plotSingleData(ayLeftAxis, timeAxis, "Left (Ay)", "Ay_Left.png")
    # plotSingleData(azLeftAxis, timeAxis, "Left (Az)", "Az_Left.png")
    # plotSingleData(gxLeftAxis, timeAxis, "Left (Gx)", "Gx_Left.png")
    # plotSingleData(gyLeftAxis, timeAxis, "Left (Gy)", "Gy_Left.png")
    # plotSingleData(gzLeftAxis, timeAxis, "Left (Gz)", "Gz_Left.png")


if __name__ == '__main__':
    main()

'''
OLD PREDICT WORKOUT

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

'''


'''
## EXAMPLE TO TRAIN THE PERFECT MODEL
def main():

    # # To hold all the slopes for a certain exercise
    # allSlopes = []
    
    workoutData = []
    # workout = readJSON("workout.json")
    # workoutData.append(workout)

    # To hold all the slopes for a certain exercise including the perfect model
    allBenchPressSlopes = []
    allCurlsSlopes = []
    allTricepSlopes = []
    
    benchPressData = []
    benchPressPerfect1 = readJSON("./Bench/bench_perfect1.json")
    benchPressData.append(benchPressPerfect1)
    benchPressPerfect2 = readJSON("./Bench/bench_perfect2.json")
    benchPressData.append(benchPressPerfect2)
    benchPressPerfect3 = readJSON("./Bench/bench_perfect3.json")
    benchPressData.append(benchPressPerfect3)
    benchPressPerfect4 = readJSON("./Bench/bench_perfect4.json")
    benchPressData.append(benchPressPerfect4)

    curlsData = []
    curlsPerfect1 = readJSON("./Curls/curls_perfect1.json")
    curlsData.append(curlsPerfect1)
    curlsPerfect2 = readJSON("./Curls/curls_perfect2.json")
    curlsData.append(curlsPerfect2)
    curlsPerfect3 = readJSON("./Curls/curls_perfect3.json")
    curlsData.append(curlsPerfect3)
    curlsPerfect4 = readJSON("./Curls/curls_perfect4.json")
    curlsData.append(curlsPerfect4)

    tricepsData = []
    tricepsPerfect1 = readJSON("./Triceps/triceps_perfect1.json")
    tricepsData.append(tricepsPerfect1)
    tricepsPerfect2 = readJSON("./Triceps/triceps_perfect2.json")
    tricepsData.append(tricepsPerfect2)
    tricepsPerfect3 = readJSON("./Triceps/triceps_perfect3.json")
    tricepsData.append(tricepsPerfect3)
    tricepsPerfect4 = readJSON("./Triceps/triceps_perfect4.json")
    tricepsData.append(tricepsPerfect4)

    allBenchPressSlopes = generateSlopes(benchPressData, "BenchPress")
    allCurlsSlopes = generateSlopes(curlsData, "Curls")
    allTricepSlopes = generateSlopes(tricepsData, "Triceps")

    # myDf1 = pd.DataFrame(benchPressData)
    # myDf1.to_csv('output_bench_press.csv', index=False, header=False)
    # myDf2 = pd.DataFrame(curlsData)
    # myDf2.to_csv('output_curls.csv', index=False, header=False)
    # myDf3 = pd.DataFrame(tricepsData)
    # myDf3.to_csv('output_triceps.csv', index=False, header=False)


## EXAMPLE TO PLOT ALL GRAPHS
def main():
    benchPressData = []
    benchPress0 = readJSON("./All/BenchAll/bench0.json")
    benchPress1 = readJSON("./All/BenchAll/bench1.json")
    benchPress2 = readJSON("./All/BenchAll/bench2.json")
    benchPress3 = readJSON("./All/BenchAll/bench3.json")
    benchPress4 = readJSON("./All/BenchAll/bench4.json")
    benchPress5 = readJSON("./All/BenchAll/bench5.json")
    benchPress6 = readJSON("./All/BenchAll/bench6.json")
    benchPress7 = readJSON("./All/BenchAll/bench7.json")
    benchPress8 = readJSON("./All/BenchAll/bench8.json")
    benchPress9 = readJSON("./All/BenchAll/bench9.json")
    benchPress10 = readJSON("./All/BenchAll/bench10.json")
    benchPressData.append(benchPress0)
    benchPressData.append(benchPress1)
    benchPressData.append(benchPress2)
    benchPressData.append(benchPress3)
    benchPressData.append(benchPress4)
    benchPressData.append(benchPress5)
    benchPressData.append(benchPress6)
    benchPressData.append(benchPress7)
    benchPressData.append(benchPress8)
    benchPressData.append(benchPress9)
    benchPressData.append(benchPress10)

    curlsData = []
    curls0 = readJSON("./All/CurlsAll/curls0.json")
    curls1 = readJSON("./All/CurlsAll/curls1.json")
    curls2 = readJSON("./All/CurlsAll/curls2.json")
    curls3 = readJSON("./All/CurlsAll/curls3.json")
    curls4 = readJSON("./All/CurlsAll/curls4.json")
    curls5 = readJSON("./All/CurlsAll/curls5.json")
    curls6 = readJSON("./All/CurlsAll/curls6.json")
    curlsData.append(curls0)
    curlsData.append(curls1)
    curlsData.append(curls2)
    curlsData.append(curls3)
    curlsData.append(curls4)
    curlsData.append(curls5)
    curlsData.append(curls6)

    tricepsData = []
    triceps0 = readJSON("./All/TricepsAll/triceps0.json")
    triceps1 = readJSON("./All/TricepsAll/triceps1.json")
    triceps2 = readJSON("./All/TricepsAll/triceps2.json")
    triceps3 = readJSON("./All/TricepsAll/triceps3.json")
    triceps4 = readJSON("./All/TricepsAll/triceps4.json")
    triceps5 = readJSON("./All/TricepsAll/triceps5.json")
    triceps6 = readJSON("./All/TricepsAll/triceps6.json")
    triceps7 = readJSON("./All/TricepsAll/triceps7.json")
    triceps8 = readJSON("./All/TricepsAll/triceps8.json")
    triceps9 = readJSON("./All/TricepsAll/triceps9.json")
    triceps10 = readJSON("./All/TricepsAll/triceps10.json")
    tricepsData.append(triceps0)
    tricepsData.append(triceps1)
    tricepsData.append(triceps2)
    tricepsData.append(triceps3)
    tricepsData.append(triceps4)
    tricepsData.append(triceps5)
    tricepsData.append(triceps6)
    tricepsData.append(triceps7)
    tricepsData.append(triceps8)
    tricepsData.append(triceps9)
    tricepsData.append(triceps10)

    allBenchPressSlopes = generateSlopes(benchPressData, "BenchPress")
    allCurlsSlopes = generateSlopes(curlsData, "Curls")
    allTricepSlopes = generateSlopes(tricepsData, "Triceps")

    myDf1 = pd.DataFrame(allBenchPressSlopes)
    myDf1.to_csv('output_bench_press.csv', index=False, header=False)
    myDf2 = pd.DataFrame(allCurlsSlopes)
    myDf2.to_csv('output_curls.csv', index=False, header=False)
    myDf3 = pd.DataFrame(allTricepSlopes)
    myDf3.to_csv('output_triceps.csv', index=False, header=False)






'''