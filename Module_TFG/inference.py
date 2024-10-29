from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
from gfpgan import GFPGANer

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--cache', default=True,
					help='Activate cache to store face detection results to eliminate face detection process from the second attempt for the same video')

parser.add_argument('--multiplier', type=int, default=1,
					help='Speed multiplier to skip face detection frames to speed up the process')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, multiplier=1):
	cache_file_name = os.path.splitext(args.face)[0] + ".txt"
	predictions = []
	if args.cache and os.path.isfile(cache_file_name):
		file = open(cache_file_name, 'r')
		for line in file:
			(x1, y1, x2, y2) = line.split(', ')
			predictions.append([int(x1), int(y1), int(x2), int(y2)])
		file.close()
		
		if len(images) != len(predictions):
			predictions = []
	
	if len(predictions) == 0:
		detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
												flip_input=False, device=device)
		batch_size = args.face_det_batch_size
		
		while 1:
			try:
				for i in tqdm(range(0, len(images), batch_size * multiplier)):
					predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size]), multiplier))
			except RuntimeError:
				if batch_size == 1: 
					raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
				batch_size //= 2
				print('Recovering from OOM error; New batch size: {}'.format(batch_size))
				continue
			break
		del detector
		
		if args.cache:
			file = open(cache_file_name, 'w')
			for (x1, y1, x2, y2) in predictions:
				file.write(str(x1) + ", " + str(y1) + ", " + str(x2) + ", " + str(y2) + "\n")
			file.close()

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, mels, multiplier):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames, multiplier) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]], multiplier)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


def main():
	# ------------------------ set up GFPGAN restorer ------------------------
	arch = 'clean'
	channel_multiplier = 2
	model_name = 'GFPGANv1.3'


	# determine model paths
	model_path = os.path.join('../GFPGAN/experiments/pretrained_models', model_name + '.pth')
	if not os.path.isfile(model_path):
		model_path = os.path.join('realesrgan/weights', model_name + '.pth')
	if not os.path.isfile(model_path):
		raise ValueError(f'Model {model_name} does not exist.')

	restorer = GFPGANer(
		model_path=model_path,
		upscale=1,
		arch=arch,
		channel_multiplier=channel_multiplier,
		bg_upsampler=None)

	# 若人脸输入文件不存在
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	# 若人脸输入文件为图片
	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps
		multiplier = 1

	# 若人脸输入文件为视频
	else:
		video_stream = cv2.VideoCapture(args.face)
		if args.fps:
			fps = args.fps
		else:
			fps = video_stream.get(cv2.CAP_PROP_FPS)
		if args.multiplier:
			multiplier = args.multiplier
		else:
			multiplier = 1
		
		# 读取视频帧图像
		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	# 打印输入人脸总帧数
	print("Number of frames available for inference(人脸文件总帧数): " + str(len(full_frames)))

	# 若音频输入文件不是wav格式，转换为wav格式
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	# 此处音频采样率默认16000
	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)	# 梅尔频谱
	print(mel.shape)

	# 若梅尔频谱中有nan，即完全空白的音段，则报错，故合成音频时需添加噪音
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	# 将梅尔频谱分块？
	mel_chunks = []
	mel_idx_multiplier = 80./fps	# 梅尔频谱块数，80是？
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	# 截取帧图像中与梅尔频谱块数相对应的部分
	full_frames = full_frames[:len(mel_chunks)]


	batch_size = args.wav2lip_batch_size
	# gen是帧图像的迭代器
	gen = datagen(full_frames.copy(), mel_chunks, multiplier)


	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			# p是一帧内生成的脸部+口型图片，在此添加修复
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			# 此时的f是p回贴进去后最终的图片，在此添加修复

			# ------------------------ restore ------------------------
			# restore faces and background if necessary
			_, _, restored_f = restorer.enhance(
				f, has_aligned=False, only_center_face=True, paste_back=True)

			out.write(restored_f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()