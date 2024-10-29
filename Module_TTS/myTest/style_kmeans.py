import torchaudio
import yaml
from munch import Munch

import utils
from data_utils import StyleLoader
from style_models import load_ASR_models, load_F0_models, build_model, load_checkpoint

from sklearn.cluster import KMeans
import torch

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

hps = utils.get_hparams()
style_dataset = StyleLoader(['angry', 'apologetic', 'excited', 'fear', 'happy', 'sad'], hps.data)

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

refs = {}
for i in range(len(style_dataset)):
    audio = style_dataset[i]
    ref_mel = preprocess(audio.numpy())
    # 防止长度过短
    if len(ref_mel[0][0][0]) < 100:
        ref_mel = ref_mel.repeat(1, 1, 1, 3)
    # print(mel.shape)
    ref = style_model.style_encoder(ref_mel).numpy()


