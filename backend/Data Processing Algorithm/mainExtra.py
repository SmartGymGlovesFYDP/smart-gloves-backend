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
## EXAMPLE TO TRAIN THE PERFECT MODEL
def main():

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


'''