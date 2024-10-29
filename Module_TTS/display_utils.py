import matplotlib.pyplot as plt
import torch
import torchaudio
import yaml
from click import style
from munch import Munch
from scipy.io.wavfile import read
import librosa.display
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from data_utils import *
import utils
from losses import discriminator_loss, kl_loss, feature_loss, generator_loss, F0_loss
from mel_processing import spec_to_mel_torch, mel_spectrogram_torch
from models import MultiPeriodDiscriminator
from style_models import load_ASR_models, build_model, load_checkpoint, load_F0_models
from text.symbols import symbols
import torch.nn.functional as F


plt.rcParams['font.family'] = 'SimHei'


def pearson_correlation(x, y):
    # 确保x和y是一维张量
    x = x.view(-1)
    y = y.view(-1)

    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算标准差
    std_x = torch.std(x)
    std_y = torch.std(y)

    # 标准化
    x_normalized = (x - mean_x) / std_x
    y_normalized = (y - mean_y) / std_y

    # 计算Pearson相关系数
    correlation = torch.mean(x_normalized * y_normalized)

    return correlation

def display_wave():
    # 音频文件路径
    file_path = 'output/new.wav'

    # 读取音频文件
    sample_rate, waveform = read(file_path)

    # 绘制波形
    plt.figure(figsize=(20, 5))
    plt.plot(waveform)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def display_mel():
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    # 加载音频文件
    # file_path = 'output/duoyin2_no_bert.wav'  # 替换为你的音频文件路径
    file_path = '../../dataset/Wave/005175.wav'
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


def display_infer_time():

    # 实验数据
    data = [
        ('短文本', 37, 0.86),
        ('长文本', 308, 1.28),
        ('含数字的文本', 87, 0.97),
        ('中英混合文本', 91, 0.93),
        ('多音字文本1', 35, 0.83),
        ('多音字文本2', 23, 0.94)
    ]

    data.sort(key=lambda x:x[1])

    # 分离文本长度和推理用时
    text_lengths = [length for _, length, _ in data]
    inference_times = [time for _, _, time in data]
    data_types = [t for t, _, _ in data]

    # 设置x轴的位置（对于柱状图，我们需要为每个柱子设置一个位置）
    x_positions = range(len(text_lengths))

    plt.ylim(None, 2)

    # 绘制柱状图
    bars = plt.bar(x_positions, inference_times, color='skyblue', alpha=0.7, label='推理用时(s)')

    # 绘制折线图
    plt.plot(x_positions, inference_times, marker='o', linestyle='-', color='red', label='数据点')

    for t, bar, value in zip(data_types, bars, inference_times):
        x = bar.get_x() + bar.get_width() / 2  # 获取柱子的中心位置
        y = value + 0.05  # 数值就是y坐标
        plt.text(x, y, t, ha='center', va='bottom')  # 在柱子上方居中显示数值

    # 添加x轴标签（这里我们使用文本长度的索引作为临时标签，你可以根据需要替换为实际长度）
    plt.xticks(x_positions, [str(length) for length in text_lengths], rotation=45, ha='right')

    # 添加标题和坐标轴标签
    plt.title('推理用时和文本长度的关系')
    plt.xlabel('文本长度(字符)')
    plt.ylabel('推理用时(s)')

    # 添加图例
    plt.legend()

    # 显示网格（可选）
    plt.grid(axis='y', linestyle='--')

    # 显示图形
    plt.tight_layout()  # 确保所有标签都可见
    plt.show()

def display_face_plot():
    import matplotlib.pyplot as plt

    # 定义数据
    groups = ['组2', '组3', '组4', '', '组5', '组6', '组7']
    inference_time = [872, 294, 7, np.NaN, 667, 308, 18]
    total_time = [979, 412, 110, np.NaN, 1386, 1013, 745]
    optimization_time = [107, 118, 103, np.NaN, 719, 705, 727]

    # 绘制推理时间和总时间的堆叠柱状图
    bar_width = 0.35
    index = range(len(groups))

    # 创建图形和主y轴
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 推理时间
    inference_bars = ax1.bar(index, inference_time, bar_width, label='推理时间', color='b')

    # 总时间（堆叠在推理时间上）
    total_bars = ax1.bar(index, [total_time[t] - i for t, i in enumerate(inference_time)], bar_width, bottom=inference_time, label='其他时间',
                         color='skyblue', edgecolor='w')
    # 注意：这里我们实际上计算的是“其他时间”，但你可以通过图例或标题来解释总时间是推理时间和其他时间的和

    # 添加次y轴用于优化时间
    ax2 = ax1.twinx()
    optimization_line, = ax2.plot(index, optimization_time, 'k--o',color='orange', label='优化时间', markersize=5)
    ax2.set_ylabel('优化时间(s)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    # ax2.spines.right.set_position(('outward', 0))  # 调整次y轴的位置以避免与主y轴刻度重叠

    # 设置主y轴标签和刻度
    ax1.set_ylabel('总时间(s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 设置图表的其他属性
    ax1.set_xlabel('组别')
    ax1.set_title('各组推理时间、优化时间和总时间对比')
    ax1.set_xticks(index)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(None, 1800)
    ax2.set_ylim(None, 1000)

    # 自定义图例（可选，如果你想要更清晰地说明总时间）
    # 注意：这里我们实际上没有直接绘制“总时间”的柱状图作为单独的数据系列
    # 但你可以通过文本描述来解释
    custom_handles = [inference_bars[0], total_bars[0], optimization_line]
    custom_labels = ['推理时间', '优化时间', '优化时间']
    ax1.legend(handles=custom_handles, labels=custom_labels, loc='upper left', bbox_to_anchor=(0.1, 0.92), ncol=1,
               borderaxespad=0.)

    cached = [False, False, True, np.NaN, False, False, True]
    multiplier = [1, 3, 1, np.NaN, 1, 3, 1]
    for c, m, bar, value in zip(cached, multiplier, total_bars, total_time):
        x = bar.get_x() + bar.get_width() / 2  # 获取柱子的中心位置
        y = value + 10 # 数值就是y坐标
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax1.text(x, y + 50, f'Cached: {c}' , ha='center', va='bottom')  # 在柱子上方居中显示数值
        ax1.text(x, y, f'Multiplier: {m}', ha='center', va='bottom')  # 在柱子上方居中显示数值

    # 或者，如果你想要直接解释总时间（尽管它在视觉上已经通过堆叠表示）
    # 你可以创建一个自定义的Patch对象来代表总时间（但这通常不是必需的）
    # ...（这里省略了创建自定义Patch对象的代码，因为它比较复杂且通常不是必需的）

    # 显示图表
    # plt.tight_layout()
    plt.show()

def loss_pitch():
    import librosa
    import numpy as np

    def extract_f0(audio, sr):
        """
        从音频信号中提取基频（F0）。
        """
        # 使用librosa的piptrack方法提取基频
        f0, voiced_flag, _ = librosa.pyin(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # 清理非声音部分的基频（通常设置为0或NaN）
        f0 = np.where(voiced_flag, f0, 0)
        return f0

    def compare_f0_loss(audio1, audio2, sr):
        """
        比较两个音频文件的每帧音高（基频）loss。
        """
        # 提取基频
        f0_1 = extract_f0(audio1, sr)
        f0_2 = extract_f0(audio2, sr)

        # 为了简化，我们假设两个音频的帧数相同或接近，或者我们只比较前N帧
        # 实际上，你可能需要更复杂的对齐策略，如动态时间规整（DTW）
        min_frames = min(len(f0_1), len(f0_2))

        # 计算每帧的音高loss（这里简单地使用绝对差异）
        f0_loss = np.abs(f0_1[:min_frames] - f0_2[:min_frames])

        return f0_loss

        # 加载音频文件

    y1, sr1 = librosa.load('output/duoyin2.wav')
    y2, sr2 = librosa.load('output/duoyin2_no_bert.wav')

    # 确保采样率相同
    if sr1 != sr2:
        sr = max(sr1, sr2)
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=sr)
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr)
    else:
        sr = sr1

        # 比较音高loss
    f0_1 = extract_f0(y1, sr)
    f0_2 = extract_f0(y2, sr)
    plt.plot(range(len(f0_1)), f0_1)
    plt.plot(range(len(f0_2)), f0_2)
    f0_loss = compare_f0_loss(y1, y2, sr)
    # plt.plot(range(len(f0_loss)), f0_loss)
    plt.grid(True)
    plt.ylim(150, 350)
    plt.show()

    # 输出或处理loss
    print("每帧的音高loss:", np.mean(f0_loss))


def revise_pitch():
    y, sr = librosa.load('output/new.wav', sr=16000)
    import soundfile as sf
    from pydub import AudioSegment
    for pitch in range(-10, 10):
        y_tuned = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch, bins_per_octave=18)
        sf.write(f'aaPitchShifter/data/pitch={pitch}.wav', y_tuned, 16000)

def pitch_extractor():
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    mean, std = -4, 4

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


    hps = utils.get_hparams()
    hps.model_dir = hps.model_dir.replace('\\', '/')

    style_dataset = StyleLoader(['angry', 'apologetic', 'excited', 'fear', 'happy', 'sad'], hps.data)
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=1,
        rank=0,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )

    net_g = utils.load_class(hps.train.train_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda()

    utils.load_model('models/vits/vits_bert_model.pth', net_g)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()

    # utils.load_model('models/vits_bert_model.pth', net_d)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # net_g = DDP(net_g)
    #
    #
    # net_d = DDP(net_d)


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    for epoch in range(10000000):
        for batch_idx, (x, x_lengths, bert, spec, spec_lengths, y, y_lengths, style) in enumerate(train_loader):
            """
                x: text_padded
                x_lengths: text_lengths
                bert: bert_padded
                spec: spec_padded
                y: wav_padded
                y_lengths: wav_lengths
            """
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(
                non_blocking=True
            )
            spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(
                non_blocking=True
            )
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(
                non_blocking=True
            )
            bert = bert.cuda(non_blocking=True)

            audio = style_dataset[batch_idx % len(style_dataset)]
            ref_mel = preprocess(audio.numpy())
            # 防止长度过短
            if len(ref_mel[0][0][0]) < 100:
                ref_mel = ref_mel.repeat(1, 1, 1, 3)
            # print(mel.shape)
            ref = style_model.style_encoder(ref_mel).cuda() * 10


            # mels = spec_to_mel_torch(
            #     spec,
            #     hps.data.filter_length,
            #     hps.data.n_mel_channels,
            #     hps.data.sampling_rate,
            #     hps.data.mel_fmin,
            #     hps.data.mel_fmax,
            # )
            #
            # ss = []
            # for bib in range(len(spec_lengths)):
            #     mel_length = int(spec_lengths[bib].item())
            #     mel = mels[bib, :, :spec_lengths[bib]]
            #     s = style_model.style_encoder(mel.unsqueeze(0).unsqueeze(1).cpu())
            #     ss.append(s)
            # ref = torch.stack(ss).squeeze().cuda()

            # with torch.no_grad():
            #     ref = style_model.style_encoder(mel.cpu()).cuda()

            with autocast(enabled=hps.train.fp16_run):
                # print(spec.transpose(1, 2).shape, ref.shape)
                # spec_fake = net_g.sp(spec.transpose(1, 2), ref).transpose(1, 2)
                # spec = spec_fake
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                    (z, z_p, z_r, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, bert, spec, spec_lengths, style=ref)
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice
                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
                    # print(f'loss_disc_all: {loss_disc_all}')
            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel - torch.mean(y_mel, dim=2, keepdim=True), y_hat_mel - torch.mean(y_hat_mel, dim=2, keepdim=True)) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    if z_r == None:
                        loss_kl_r = 0
                    else:
                        loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)

                    # TODO: 这里加上计算生成的音频的style_loss，不过似乎不太行，试了几个预测出来的值（y_hat, y_hat_mel...）及其变换，都无法正常编码。。。
                    # 已经解决，原因是生成的y_mel长度过短，在style_encoder卷积中出现不适配，手动repeat对其扩充即可
                    # ss = []
                    # for bib in range(len(spec_lengths)):
                    #     mel_length = int(spec_lengths[bib].item())
                    #     mel = y_mel[bib, :, :spec_lengths[bib]]
                    #     s = style_model.style_encoder(mel.unsqueeze(0).unsqueeze(1).cpu())
                    #     ss.append(s)
                    #
                    # style_fake = torch.stack(ss).squeeze().cuda()
                    mel_fake = y_hat_mel.repeat(1, 1, 2).cpu()
                    style_fake = style_model.style_encoder(mel_fake.unsqueeze(1)).cuda()
                    loss_style = torch.nn.MSELoss()(ref.repeat(hps.train.batch_size, 1), style_fake)


                    F0_real, _, _ = e(ref_mel.cpu())
                    F0_real = torch.mean(F0_real, dim=0, keepdim=True).unsqueeze(0).repeat(hps.train.batch_size, 1)
                    # print(F0_real.shape)
                    F0_fake, _, _ = e(y_hat_mel.unsqueeze(1).cpu())
                    # print(F0_fake.shape)
                    F0_fake = torch.mean(F0_fake, dim=1, keepdim=True).squeeze(0)
                    # print(F0_fake.shape)
                    loss_F0 = F.smooth_l1_loss(F0_real, F0_fake)



                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_kl_r + loss_F0 * 0.05 + loss_style * 10
                    # loss_gen_all = loss_F0 * 0.5 + loss_style * 2 + loss_fm * 0.5
                    loss_gen_all_print = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_kl_r + loss_F0 + loss_style


                    # print(y_hat_mel.shape)

                    # print(f'loss_gen_all: {loss_gen_all}')
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()


            print(f'loss_d:{loss_disc_all:.2f}, loss_g:{loss_gen_all:.2f}, loss_mel:{loss_mel:.2f}, loss_F0:{loss_F0:.2f}, loss_style:{loss_style:.2f}, loss_fm: {loss_fm:.2f}')
        print(f'epoch{epoch}: loss_d:{loss_disc_all}, loss_g:{loss_gen_all}')


pitch_extractor()