import os
import sys
import time

import librosa
import numpy
import numpy as np

import torch
import yaml
from munch import Munch

import utils
import argparse

from scipy.io import wavfile

from style_models import load_ASR_models, build_model, load_F0_models, load_checkpoint
from text.symbols import symbols
from text import cleaned_text_to_sequence
from train import to_mel, mean, std
from vits_pinyin import VITS_PinYin

from pydub import AudioSegment

parser = argparse.ArgumentParser(description='Inference code for bert vits models')
parser.add_argument('--config', type=str, default="configs/bert_vits.json")
parser.add_argument('--model', type=str, default="models/vits_bert_model.pth")
args = parser.parse_args()

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))
    sound = AudioSegment.from_file(path, format="wav")
    sound.set_frame_rate(24000).set_sample_width(4).set_channels(1).export(path, format="wav")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# pinyin
tts_front = VITS_PinYin("./bert", device)

# config
hps = utils.get_hparams_from_file(args.config)

# model
net_g = utils.load_class(hps.train.eval_class)(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)

# model_path = "logs/bert_vits/G_200000.pth"
# utils.save_model(net_g, "vits_bert_model.pth")
# model_path = "vits_bert_model.pth"
steps = 2500
args.model = f'logs/style_vits/G_{steps}.pth'
# args.model = 'models/vits/vits_bert_model.pth'
utils.load_model(args.model, net_g)
net_g.eval()
net_g.to(device)

os.makedirs("./output/", exist_ok=True)

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

if __name__ == "__main__":
    # n = 0
    # fo = open("vits_infer_item.txt", "r+", encoding='utf-8')
    # while (True):
    #     try:
    #         item = fo.readline().strip()
    #     except Exception as e:
    #         print('nothing of except:', e)
    #         break
    #     if (item == None or item == ""):
    #         break
    #     n = n + 1
    #     phonemes, char_embeds = tts_front.chinese_to_phonemes(item)
    #     input_ids = cleaned_text_to_sequence(phonemes)
    #     with torch.no_grad():
    #         x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    #         x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
    #         x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
    #         audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
    #                             length_scale=1)[0][0, 0].data.cpu().float().numpy()
    #     save_wav(audio, f"./output/bert_vits_{n}.wav", hps.data.sampling_rate)
    # fo.close()
    # load StyleTTS
    # model_path = "./Models/LJSpeech/epoch_2nd_00180.pth"
    model_config_path = "models/LJSpeech/config.yml"

    config = yaml.safe_load(open(model_config_path))

    # load pretrained ASR model
    text_aligner = load_ASR_models('Utils/ASR/epoch_00080.pth', 'Utils/ASR/config.yml')

    # load pretrained F0 model
    e = load_F0_models('Utils/JDC/bst.t7')

    style_model = build_model(Munch(config['model_params']), text_aligner, e)
    load_checkpoint(style_model, None, 'models/LJSpeech/epoch_2nd_00180.pth')




    start = time.time()

    emotion = 'Sad'
    wave, sr = librosa.load(f'test_data/{emotion}.wav', sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    try:
        with torch.no_grad():
            # print(mel_tensor.unsqueeze(1).shape)
            # print(mel_tensor.unsqueeze(1).shape)
            ref = style_model.style_encoder(mel_tensor.unsqueeze(1).cpu()).cuda() * 10
            print(ref)
    except Exception as e:
        print(e)

    text = "华中科技大学是一所非常好的学校"
    # text = "天津人"
    phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
    print(f'输入文本——>\n{text}\n生成韵律——>\n{phonemes}')
    input_ids = cleaned_text_to_sequence(phonemes)
    # print(phonemes, input_ids)
    # exit()
    with torch.no_grad():
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.667,
                            length_scale=1, style=ref)[0][0, 0].data.cpu().float().numpy()

    # with torch.no_grad():
    #     x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    #     x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
    #     audio = net_g.infer(x_tst, x_tst_lengths, bert=None, noise_scale=0.5,
    #                         length_scale=1)[0][0, 0].data.cpu().float().numpy()
    if len(text) > 10:
        text = text[:9]
    save_path = f"./output/{steps}_{emotion}.wav"
    save_wav(audio, save_path, hps.data.sampling_rate)
    print(f"语音文件已保存至 {steps}_{emotion}")
    end = time.time()
    print(f'运行用时：{end - start:.2f}s')

    # import pygame
    # pygame.mixer.init()
    # pygame.mixer.music.load(os.path.abspath(save_path))
    # pygame.mixer.music.play()
    # while pygame.mixer.music.get_busy() == True:
    #     continue
    # pygame.mixer.music.unload()
    #
    # os.remove(os.path.abspath(save_path))

