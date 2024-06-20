# Measure pitch of all wav files in directory
import glob
import numpy as np
import pandas as pd
import parselmouth
import os
import parselmouth.praat as praat
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID)  # read the sound

    pitch = sound.to_pitch()  # create a praat pitch object
    pitch_values = pitch.selected_array['frequency']
    max_pitch = np.max(pitch_values)  # Maximum pitch
    mean_pitch = call(pitch, "Get mean", 0, 0, unit)  # mean pitch
    min_pitch = np.min(pitch_values)  # Minimum pitch
    std_of_pitch = call(pitch, "Get standard deviation", 0, 0, unit)  # standard deviation of the pitch

    pointProcess = call([sound, pitch], "To PointProcess (cc)")
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)  # five jitter
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    intensity = sound.to_intensity()
    mean_intensity = intensity.values.mean()  # mean intensity
    # Extract minimum intensity
    min_intensity = intensity.values.min()  # minimum intensity
    # Extract maximum intensity
    max_intensity = intensity.values.max()  # maximum intensity

    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,
                        1.6)  # six instability measures
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    voice_report = call([sound, pitch, pointProcess], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45).split(
        chr(10))
    number_of_pulses = int(voice_report[8].split(":")[-1])  # Number of pulses

    number_of_periods = int(voice_report[9].split(":")[-1])  # Number of periods
    mean_period = voice_report[10].split(":")[-1].strip().split(" ")[0]  # mean period
    stdev_period = voice_report[11].split(":")[-1].strip().split(" ")[0]  # standard deviation of the period

    fraction_of_breaks = voice_report[13].split(":")[-1].strip().split("(")[0]  # Fraction of breaks
    number_of_breaks = int(voice_report[14].split(":")[-1].strip().split(" ")[0])  # number of breaks
    degree_of_breaks = voice_report[15].split(":")[-1].strip().split("(")[0]  # degree of breaks

    mean_n_h_ratio = voice_report[-3].split(":")[-1]  # Mean noise/harmonics ratio
    mean_h_n_ratio = voice_report[-2].split(":")[-1].strip().split(" ")[0]  # Mean harmonics/noise ratio
    autocorrelation = voice_report[-4].split(":")[-1]  # Autocorrelation

    return max_pitch, mean_pitch, min_pitch, std_of_pitch, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, mean_intensity, min_intensity, max_intensity, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, number_of_pulses, number_of_periods, mean_period, stdev_period, fraction_of_breaks, number_of_breaks, degree_of_breaks, mean_n_h_ratio, mean_h_n_ratio, autocorrelation
    # return max_pitch, mean_pitch, min_pitch, std_of_pitch, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


def runPCA(df):
    # Z-score the Jitter and Shimmer measurements
    features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    # y = df.loc[:,['target']].values
    # Standardizing the features
    #     x = ['0.011564539382372126' '0.00011747650513271417' '0.0027043170159623403'
    #   '0.000839601601570182' '0.008112951047887021' '0.10961658651211333'
    #   '0.9780445404346173' '0.03823243259460102' '0.10964921505403936' 'nan'
    #   '0.11469729778380305']
    x = x[x != 'nan']
    #      = np.where(arr == value_to_replace, replacement_value, arr)
    print(x)
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['JitterPCA', 'ShimmerPCA'])
    return principalDf


# path = "E:/year3_sem1/SA/video_image/split/transcript/audio/3000141124/SPEAKER_01"
# path = "E:/year3_sem2/SA/video_prediction/Transcript/audio/3000141124/not_sure"
def audio_features_extractor(path):
    # create lists to put the results
    file_list = []
    max_pitch_lst = []
    mean_pitch_lst = []
    min_pitch_lst = []
    std_of_pitch_lst = []

    localJitter_lst = []
    localabsoluteJitter_lst = []
    rapJitter_lst = []
    ppq5Jitter_lst = []
    ddpJitter_lst = []

    mean_intensity_lst = []
    # Extract minimum intensity
    min_intensity_lst = []
    # Extract maximum intensity
    max_intensity_lst = []

    localShimmer_lst = []
    localdbShimmer_lst = []
    apq3Shimmer_lst = []
    aqpq5Shimmer_lst = []
    apq11Shimmer_lst = []
    ddaShimmer_lst = []

    num_of_pulses_lst = []

    num_of_periods_lst = []
    mean_period_lst = []
    std_period_lst = []  # standard deviation of the period

    fraction_of_breaks_lst = []
    number_of_breaks_lst = []
    degree_of_breaks_lst = []

    mean_n_h_ratio_lst = []
    mean_h_n_ratio_lst = []

    autocorrelation_lst = []

    lst_lst = []

    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob(os.path.join(path, '*.wav')):
        sound = parselmouth.Sound(wave_file)
        max_pitch, mean_pitch, min_pitch, std_of_pitch, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, mean_intensity, min_intensity, max_intensity, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, number_of_pulses, number_of_periods, mean_period, stdev_period, fraction_of_breaks, number_of_breaks, degree_of_breaks, mean_n_h_ratio, mean_h_n_ratio, autocorrelation = measurePitch(
            sound, 75, 500, "Hertz")
        #     for i in [max_pitch, mean_pitch, min_pitch, std_of_pitch, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, mean_intensity, min_intensity, max_intensity, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, hnr]:
        #         if  np.isnan(i):
        #             i = 0
        file_list.append(os.path.splitext((wave_file.split("/")[-1]).split('\\')[-1])[0])  # make an ID list
        max_pitch_lst.append(max_pitch)
        mean_pitch_lst.append(mean_pitch)
        min_pitch_lst.append(min_pitch)
        std_of_pitch_lst.append(std_of_pitch)
        localJitter_lst.append(localJitter)
        localabsoluteJitter_lst.append(localabsoluteJitter)
        rapJitter_lst.append(rapJitter)
        ppq5Jitter_lst.append(ppq5Jitter)
        ddpJitter_lst.append(ddpJitter)
        mean_intensity_lst.append(mean_intensity)
        min_intensity_lst.append(min_intensity)
        max_intensity_lst.append(max_intensity)
        localShimmer_lst.append(localShimmer)
        localdbShimmer_lst.append(localdbShimmer)
        apq3Shimmer_lst.append(apq3Shimmer)
        aqpq5Shimmer_lst.append(aqpq5Shimmer)
        apq11Shimmer_lst.append(apq11Shimmer)
        ddaShimmer_lst.append(ddaShimmer)
        num_of_pulses_lst.append(number_of_pulses)
        num_of_periods_lst.append(number_of_periods)
        mean_period_lst.append(mean_period)
        std_period_lst.append(stdev_period)  # standard deviation of the period
        fraction_of_breaks_lst.append(fraction_of_breaks)
        number_of_breaks_lst.append(number_of_breaks)
        degree_of_breaks_lst.append(degree_of_breaks)
        mean_n_h_ratio_lst.append(mean_n_h_ratio)
        mean_h_n_ratio_lst.append(mean_h_n_ratio)
        autocorrelation_lst.append(autocorrelation)

    audio_df = pd.DataFrame(np.column_stack(
        [max_pitch_lst, mean_pitch_lst, min_pitch_lst, std_of_pitch_lst, localJitter_lst,
         localabsoluteJitter_lst, rapJitter_lst, ppq5Jitter_lst, ddpJitter_lst, mean_intensity_lst, min_intensity_lst,
         max_intensity_lst, localShimmer_lst, localdbShimmer_lst, apq3Shimmer_lst, aqpq5Shimmer_lst, apq11Shimmer_lst,
         ddaShimmer_lst, num_of_pulses_lst, num_of_periods_lst, mean_period_lst, std_period_lst, fraction_of_breaks_lst,
         number_of_breaks_lst, degree_of_breaks_lst, mean_n_h_ratio_lst, mean_h_n_ratio_lst, autocorrelation_lst]),
        columns=['max_pitch', 'mean_pitch', 'min_pitch', 'std_of_pitch', 'localJitter',
                 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'mean_intensity',
                 'min_intensity', 'max_intensity', 'localShimmer', 'localdbShimmer', 'apq3Shimmer',
                 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'number_of_pulses', 'number_of_periods',
                 'mean_period', 'stdev_period', 'fraction_of_breaks', 'number_of_breaks', 'degree_of_breaks',
                 'mean_n_h_ratio', 'mean_h_n_ratio',
                 'autocorrelation'], index=file_list)  # add these lists to pandas in the right order
    # pcaData = runPCA(df)

    return audio_df

