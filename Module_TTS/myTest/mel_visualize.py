import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


def display_mel(file_path):

    # 加载音频文件
    # file_path = 'output/duoyin2_no_bert.wav'  # 替换为你的音频文件路径
    # file_path = '../output/1500_Sad.wav'
    y, sr = librosa.load(file_path, sr=16000)  # 使用原始采样率

    # 计算梅尔频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # 转换为对数刻度
    log_S = librosa.power_to_db(S, ref=np.max)

    # 绘制梅尔声谱图
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='frames', y_axis='mel')  # 注意x_axis='frames'
    plt.colorbar(format='%+2.0f dB')
    plt.title('梅尔声谱图')
    plt.xlabel('帧数')  # 横轴标签可以是'Frame'或'Frame Index'，但通常没有具体的索引数字
    plt.ylabel('梅尔频率')
    # plt.axis('off')
    plt.tight_layout()

    # 显示图形
    plt.show()

def display_wave(file_path):
    # 音频文件路径
    # file_path = 'output/new.wav'

    # 读取音频文件
    sample_rate, waveform = read(file_path)

    # 绘制波形
    plt.figure(figsize=(20, 5))
    plt.plot(waveform)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
display_wave('../output/3000_Sad.wav')
display_wave('../output/3000_Excited.wav')