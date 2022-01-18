import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
import librosa.display
import re
import os
import time
import warnings
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy import signal as sg

### v 0.9.3 stereo version ###

#### target : build a simple daw in python（ may be used in Automatic Mixing :)）###

### 需要解决的问题：

#1.需要一个可听化的function(to do)

#2.加载更多的音频特征(to do)

#3.如何实现M/S信号分离(to do)

#4.像pydub一样切割音频，并具有淡入淡出功能(待完善)

#5.soundfile的左右声道居然没有反(done)

#6.支持16/24bit输出(done)

### basic functions（可以加入更多的音频特征）###

def dbfs(amp,bit=16):
	bit_depth = 2 ** bit
	sample = int(bit_depth*amp)
	dbfs = round(20*np.log10(sample/bit_depth),2)
	return dbfs

def dbfs_to_amp(dbfs,bit=16):
	bit_depth = 2 ** bit
	amp = math.pow(10,(dbfs/20))
	return round(amp,9)

#normalization
def MaxMinNormalization(x,i=0.5):
	Max = 0
	Min = 0
	Max = max(x)
	x = x  / Max;
	x *= i
	return x;
 
#load tracks 加载音轨
def load_track(path,filename):
	yIn, Fs = sf.read(path+'/'+filename)

	if yIn.T.shape[0] == 2:
		signal_type = 'stereo'
	elif yIn.T.shape[0] != 2:
		signal_type = 'mono'
	if signal_type == 'stereo':
		track = stereo_track(yIn, Fs, signal_type, filename)
	elif signal_type == 'mono':
		track = mono_track(yIn, Fs, signal_type, filename)
	return track

#group 合并音轨（使用这个的前提是音轨已经对齐好了）
def group(a,b,name):
	if a.signal_type == 'mono' and b.signal_type == 'mono':
		len1 = a.yIn.shape[0]
		len2 = b.yIn.shape[0]
		nframe = max(len1, len2)
		if len1 > len2:
			zero_num = int(nframe - len2)
			zero_pad = np.zeros(zero_num)
			b.yIn = np.hstack([b.yIn,zero_pad])
		elif len1 < len2:
			zero_num = int(nframe - len1)
			zero_pad = np.zeros(zero_num)
			a.yIn = np.hstack([a.yIn,zero_pad])
		mixed_y = a.yIn + b.yIn
		group_track = mono_track(mixed_y, a.Fs, signal_type='mono', filename=name)

	elif a.signal_type == 'stereo' and b.signal_type == 'mono':
		print('wrong ! please tranfrom mono to stereo( or stereo to mono)')

	elif a.signal_type == 'stereo' and b.signal_type == 'stereo':
		len1 = len(a.left)
		len2 = len(b.left)
		nframe = max(len1, len2)
		if len1 > len2:
			zero_num = int(nframe - len2)
			zero_pad = np.zeros(zero_num)
			zero_pad = np.array(zero_pad.reshape(zero_num,1))
			left_zeros = np.vstack([b.left,zero_pad])
			right_zeros = np.vstack([b.right,zero_pad])
			left_channel = a.left + left_zeros
			right_channel = a.right + right_zeros
			left_channel = np.array(left_channel.reshape(len1,1))
			right_channel = np.array(right_channel.reshape(len1,1))
			mixed_channel = np.hstack([left_channel,right_channel])

		elif len1 < len2:
			zero_num = int(nframe - len1)
			zero_pad = np.zeros(zero_num)
			zero_pad = np.array(zero_pad.reshape(zero_num,1))
			left_zeros = np.vstack([a.left,zero_pad])
			right_zeros = np.vstack([a.right,zero_pad])
			left_channel = left_zeros + b.left
			right_channel = right_zeros + b.right
			left_channel = np.array(left_channel.reshape(len2,1))
			right_channel = np.array(right_channel.reshape(len2,1))
			mixed_channel = np.hstack([left_channel,right_channel])

		else:
			left_channel = a.left + b.left
			right_channel = a.right + b.right
			left_channel = np.array(left_channel.reshape(len(a.left),1))
			right_channel = np.array(right_channel.reshape(len(a.right),1))
			mixed_channel = np.hstack([left_channel,right_channel])
		
		group_track = stereo_track(mixed_channel, a.Fs, signal_type='stereo', filename=name)

	return group_track

#cut with crossfade
def cut(a, time, crossfade=True, cross_time=200):#time(input s)cross_time(input ms)
	sr = a.Fs
	cut_frame = np.floor(time*1000*(sr/1000))# 输入为s，但精度为ms

	if a.signal_type == 'mono':

		if crossfade==True:
			cross_frame = int(np.floor(cross_time*(sr/1000)))
			cut_front = a.yIn[:int(cut_frame+cross_frame)]
			cut_back = a.yIn[int(cut_frame-cross_frame):]
			zero2one = np.linspace(0,1,2*cross_frame)
			zero_pad = np.ones(int(cut_front.shape[0]-2*cross_frame)) #zero pad for cut_front
			zero_pad2 = np.ones(int(cut_back.shape[0]-2*cross_frame)) #zero pad for cut_back
			front_pad = np.hstack([zero_pad,np.linspace(1,0,2*cross_frame)])
			back_pad = np.hstack([zero2one,zero_pad2])
			cut_front = cut_front * front_pad
			cut_back = cut_back * back_pad
			cut1 = mono_track(cut_front, a.Fs, signal_type='mono', filename=a.name[:-4]+'-1'+'.wav')
			cut2 = mono_track(cut_back, a.Fs, signal_type='mono', filename=a.name[:-4]+'-2'+'.wav')
		else:
			cut_front = a.yIn[:cut_frame]
			cut_back = a.yIn[cut_frame:]
			cut1 = mono_track(cut_front, a.Fs, signal_type='mono', filename=a.name[:-4]+'-1'+'.wav')
			cut2 = mono_track(cut_back, a.Fs, signal_type='mono', filename=a.name[:-4]+'-2'+'.wav')

	if a.signal_type == 'stereo':

		if crossfade==True:
			cross_frame = int(np.floor(cross_time*(sr/1000)))
			cut_front_L = a.left[:int(cut_frame+cross_frame)]
			cut_front_R = a.right[:int(cut_frame+cross_frame)]
			cut_back_L = a.left[int(cut_frame-cross_frame):]
			cut_back_R = a.right[int(cut_frame-cross_frame):]

			zero2one = np.linspace(0,1,2*cross_frame)
			zero_pad = np.ones(int(cut_front_L.shape[0]-2*cross_frame)) #zero pad for cut_front
			zero_pad2 = np.ones(int(cut_back_L.shape[0]-2*cross_frame)) #zero pad for cut_back
			front_pad = np.hstack([zero_pad,np.linspace(1,0,2*cross_frame)])
			front_pad = np.array(front_pad.reshape(len(front_pad),1))
			back_pad = np.hstack([zero2one,zero_pad2])
			back_pad = np.array(back_pad.reshape(len(back_pad),1))

			cut_front_L = cut_front_L * front_pad
			cut_front_R = cut_front_R * front_pad
			cut_back_L = cut_back_L * back_pad
			cut_back_R = cut_back_R * back_pad

			cut_front = np.hstack([cut_front_L,cut_front_R])
			cut_back = np.hstack([cut_back_L,cut_back_R])

			cut1 = stereo_track(cut_front, a.Fs, signal_type='stereo', filename=a.name[:-4]+'-1'+'.wav')
			cut2 = stereo_track(cut_back, a.Fs, signal_type='stereo', filename=a.name[:-4]+'-2'+'.wav')

		else:
			cut_front_L = a.left[:cut_frame]
			cut_front_R = a.right[:cut_frame]
			cut_back_L = a.left[cut_frame:]
			cut_back_R = a.right[cut_frame:]
			cut_front_L = np.array(cut_front_L.reshape(len(cut_front_L),1))
			cut_front_R = np.array(cut_front_R.reshape(len(cut_front_R),1))
			cut_back_L = np.array(cut_back_L.reshape(len(cut_back_L),1))
			cut_back_R = np.array(cut_back_R.reshape(len(cut_back_R),1))
			cut_front = np.hstack([cut_front_L,cut_front_R])
			cut_back = np.hstack([cut_back_L,cut_back_R])
			cut1 = stereo_track(cut_front, a.Fs, signal_type='stereo', filename=a.name[:-4]+'-1'+'.wav')
			cut2 = stereo_track(cut_back, a.Fs, signal_type='stereo', filename=a.name[:-4]+'-2'+'.wav')

	return cut1,cut2

#splice 拼接音轨（和group不一样,这个直接接在后面）
def splice(a, b, crossfade=True, cross_time=200):
	if a.Fs == b.Fs:
		if a.signal_type == 'mono':

			if crossfade == True:
				cross_frame = int(np.floor(cross_time*(a.Fs/1000)))
				zero_pad = np.zeros(int(b.yIn.shape[0]-2*cross_frame))
				zero_pad2 = np.zeros(int(a.yIn.shape[0]-2*cross_frame))
				front_part = np.hstack([a.yIn,zero_pad])
				back_part = np.hstack([zero_pad2,b.yIn])
				total_part = front_part+back_part
				total = mono_track(total_part,a.Fs, signal_type='mono', filename=a.name[:-4]+'-spliced'+'.wav')
			elif crossfade != True:
				total_part = np.hstack([a.yIn,b.yIn])
				total = mono_track(total_part,a.Fs, signal_type='mono', filename=a.name[:-4]+'-spliced'+'.wav')

		if a.signal_type == 'stereo':

			if crossfade == True:
				cross_frame = int(np.floor(cross_time*(a.Fs/1000)))
				zero_pad = np.zeros(int(b.left.shape[0]-2*cross_frame))
				zero_pad2 = np.zeros(int(a.left.shape[0]-2*cross_frame))
				zero_pad = np.array(zero_pad.reshape(len(zero_pad),1))
				zero_pad2 = np.array(zero_pad2.reshape(len(zero_pad2),1))

				front_part_L = np.vstack([a.left,zero_pad])
				front_part_R = np.vstack([a.right,zero_pad])
				back_part_L = np.vstack([zero_pad2,b.left])
				back_part_R = np.vstack([zero_pad2,b.right])

				front_part = np.hstack([front_part_L,front_part_R])
				back_part = np.hstack([back_part_L,back_part_R])
				total_part = front_part+back_part

				total = stereo_track(total_part,a.Fs, signal_type='stereo', filename=a.name[:-4]+'-spliced'+'.wav')
			elif crossfade != True:
				total_part_L = np.vstack([a.left,b.left])
				total_part_R = np.vstack([a.right,b.right])
				total_part = np.hstack([total_part_L,total_part_R]) 

				total = stereo_track(total_part,a.Fs, signal_type='stereo', filename=a.name[:-4]+'-spliced'+'.wav')
	else:
		print('wrong!different sample rate!')
	return total

#level adjustment
def lufs_normalize(ys,lufs,target_lufs):
	y_lufs = pyln.normalize.loudness(ys,lufs,target_lufs)
	return y_lufs


#export 导出
def export(savepath,filename,yIn,Fs):
	sf.write(savepath+'/'+filename,yIn,Fs)

#play 播放音频

### detection ###

#rms检测
def RMS_detection(ys,thre=0.001):
	idx = np.where(np.abs(ys) >= thre)
	y_signal = ys[idx]
	rms_signal = np.round(20 * np.log10((y_signal ** 2).mean() ** 0.5), 2)
	return rms_signal

#lufs检测
def LUFS_detection(ys,sr,thre=0.001):
	idx = np.where(np.abs(ys) >= thre)
	y_signal = ys[idx]
	meter = pyln.Meter(sr)
	if ys.shape[0] > sr:
		lufs_signal = np.round(meter.integrated_loudness(y_signal), 2)
	elif ys.shape[0] <= sr:
		lufs_signal = 0
	return lufs_signal

#峰值电平检测
def Peak_detection(ys,thre=0.001):
	idx = np.where(np.abs(ys) >= thre)
	y_signal = ys[idx]
	peak = np.max(y_signal)
	peak_dB = np.round(20 * np.log10((peak ** 2).mean() ** 0.5), 2)
	return peak_dB

###sound effects###

#equalizer
def low_pass_filter(ys,f0, Q=1., fs=192000):
	"""
	根据PEQ参数设计二阶IIR数字低通滤波器，默认采样率192k
	:param f0: 中心频率
	:param Q: 峰值带宽
	:param fs: 系统采样率
	:return: 双二阶滤波器系数
	"""
	w0 = 2 * np.pi * f0 / fs
	alpha = np.sin(w0) / (2 * Q)

	b0 = (1 - np.cos(w0)) / 2
	b1 = 1 - np.cos(w0)
	b2 = (1 - np.cos(w0)) / 2
	a0 = 1 + alpha
	a1 = -2 * np.cos(w0)
	a2 = 1 - alpha
	b = np.array([b0, b1, b2])
	a = np.array([a0, a1, a2])
	h = np.hstack((b / a[0], a / a[0]))
	d = h[:int(len(h)/2)]
	s = h[int(len(h)/2):]
	y = sg.filtfilt(d,s,ys)

	return y,fs

def high_pass_filter(ys,f0, Q=1.,fs=192000):

	"""
	根据PEQ参数设计二阶IIR数字高通滤波器，默认采样率192k
	:param f0: 中心频率
	:param Q: 峰值带宽
	:param fs: 系统采样率
	:return: 双二阶滤波器系数
	"""
	w0 = 2 * np.pi * f0 / fs
	alpha = np.sin(w0) / (2 * Q)

	b0 = (1 + np.cos(w0)) / 2
	b1 = -1 - np.cos(w0)
	b2 = (1 + np.cos(w0)) / 2
	a0 = 1 + alpha
	a1 = -2 * np.cos(w0)
	a2 = 1 - alpha
	b = np.array([b0, b1, b2])
	a = np.array([a0, a1, a2])

	h = np.hstack((b / a[0], a / a[0]))
	d = h[:int(len(h)/2)]
	s = h[int(len(h)/2):]
	y = sg.filtfilt(d,s,ys)

	return y,fs

def low_shelf_filter(ys,f0, gain=0., Q=1., fs=192000):
	"""
	根据PEQ参数设计二阶IIR数字low shelf滤波器，默认采样率192k
	:param f0: 中心频率
	:param gain: 峰值增益
	:param Q: 峰值带宽
	:param fs: 系统采样率
	:return: 双二阶滤波器系数
	"""
	A = np.sqrt(10 ** (gain / 20))
	w0 = 2 * np.pi * f0 / fs
	alpha = np.sin(w0) / (2 * Q)

	b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
	b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
	b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
	a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
	a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
	a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

	b = np.array([b0, b1, b2])
	a = np.array([a0, a1, a2])

	h = np.hstack((b / a[0], a / a[0]))
	d = h[:int(len(h)/2)]
	s = h[int(len(h)/2):]
	y = sg.filtfilt(d,s,ys)

	return y,fs


def high_shelf_filter(ys,f0, gain=0., Q=1., fs=192000):
	"""
	根据PEQ参数设计二阶IIR数字high shelf滤波器，默认采样率192k
	:param f0: 中心频率
	:param gain: 峰值增益
	:param Q: 峰值带宽
	:param fs: 系统采样率
	:return: 双二阶滤波器系数
	"""
	A = np.sqrt(10 ** (gain / 20))
	w0 = 2 * np.pi * f0 / fs
	alpha = np.sin(w0) / (2 * Q)

	b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
	b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
	b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
	a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
	a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
	a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

	b = np.array([b0, b1, b2])
	a = np.array([a0, a1, a2])

	h = np.hstack((b / a[0], a / a[0]))
	d = h[:int(len(h)/2)]
	s = h[int(len(h)/2):]

	y = sg.filtfilt(d,s,ys)

	return y,fs


def peak_filter(ys,f0, gain=0., Q=1.,fs=192000):
	"""
	根据PEQ参数设计二阶IIR数字peak滤波器，默认采样率192k
	:param f0: 中心频率
	:param gain: 峰值增益，正值为peak filter,负值为notch filter
	:param Q: 峰值带宽
	:param fs: 系统采样率
	:return: 双二阶滤波器系数
	"""
	A = np.sqrt(10 ** (gain / 20))
	w0 = 2 * np.pi * f0 / fs
	alpha = np.sin(w0) / (2 * Q)

	b0 = 1 + alpha * A
	b1 = -2 * np.cos(w0)
	b2 = 1 - alpha * A
	a0 = 1 + alpha / A
	a1 = -2 * np.cos(w0)
	a2 = 1 - alpha / A
	b = np.array([b0, b1, b2])
	a = np.array([a0, a1, a2])

	h = np.hstack((b / a[0], a / a[0]))
	d = h[:int(len(h)/2)]
	s = h[int(len(h)/2):]
	y = sg.filtfilt(d,s,ys)

	return y,fs

#convolution reverb
def audio_effect(sig, ir_sig, sr, seg_len = 512):
	
	n_c = int(np.ceil(len(sig) / seg_len))
	block = np.zeros([seg_len, n_c])
	

	for i in range(0, n_c):

		st = i * seg_len
		en = st + seg_len

		if en > len(sig):
			en = len(sig)

		block[0:len(sig[st:en]), i] = sig[st:en]

	block_pad = np.zeros([seg_len + len(ir_sig) - 1, n_c])
	block_pad[0:seg_len, :] = block
	ir_pad = np.zeros(len(ir_sig) + seg_len - 1)
	ir_pad[0:len(ir_sig)] = ir_sig

	fft_ir = np.fft.fft(ir_pad)
	out_sig = np.zeros(n_c * seg_len + len(ir_sig) - 1) #mono

	for m in range(0, n_c):
		fft_sig = np.fft.fft(block_pad[:, m])
		multi = fft_sig * fft_ir
		conv = np.fft.ifft(multi)
		real_conv = np.real(conv)
		st = m * seg_len
		en = st + seg_len + len(ir_sig) - 1
		out_sig[st:en] += real_conv
		
	sig_norm = out_sig * 0.7 / np.max(np.abs(out_sig))
	
	return sig_norm

#compressor

### type of the track ###

class mono_track:
	"""type for mono signal input"""
	def __init__(self, yIn,  Fs, signal_type, filename):
		self.name = str(filename)
		self.yIn = np.asanyarray(yIn)
		self.Fs = Fs
		self.signal_type = signal_type
		self.channel = 1
		self.lufs = LUFS_detection(yIn,Fs)
		self.rms = RMS_detection(yIn)
		self.peak = Peak_detection(yIn)

	###   基本功能
	def adjust_level(self,target): # 调节电平
		self.yIn = lufs_normalize(self.yIn,self.lufs,target_lufs=target)
		self.lufs = LUFS_detection(self.yIn,sr=self.Fs)
		self.rms = RMS_detection(self.yIn)
		self.peak = Peak_detection(self.yIn)
		if self.peak >= 0:
			print('warning:')
		return self.yIn, self.lufs , self.rms, self.peak

	def change_name(self,new_name):
		self.name = str(new_name)
		return self.name

	def make_spectrum(self):         ###显示不出来，需要调试###
		fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
		D = librosa.amplitude_to_db(np.abs(librosa.stft(self.yIn)), ref=np.max)
		img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
								   sr=self.Fs, ax=ax[0]) 
		ax[0].set(title='Linear spectrogram')
		ax[0].label_outer()
		hop_length = 1024
		D = librosa.amplitude_to_db(np.abs(librosa.stft(self.yIn, hop_length=hop_length)),
								ref=np.max)			 
		librosa.display.specshow(D, y_axis='log', sr=self.Fs, hop_length=hop_length,x_axis='time', ax=ax[1])                        
		ax[1].set(title='Log-frequency power spectrogram')
		ax[1].label_outer()
		fig.colorbar(img, ax=ax, format="%+2.f dB")
		plt.show()

	def normalize(self,i=0.5):
		self.yIn = MaxMinNormalization(self.yIn,i)
		self.lufs = LUFS_detection(self.yIn,sr=self.Fs)
		self.rms = RMS_detection(self.yIn)
		self.peak = Peak_detection(self.yIn)
		return self.yIn, self.lufs , self.rms, self.peak

	def copy(self):
		copy_track = mono_track(self.yIn, self.Fs, self.signal_type, self.name[:-4]+'-2'+'.wav')
		return copy_track

	def export(self,savepath):
		sf.write(savepath+'/'+self.name,self.yIn,self.Fs,'PCM_16')

	def export24(self,savepath):
		sf.write(savepath+'/'+self.name,self.yIn,self.Fs,'PCM_24')

	def mono2stereo(self):
		stereo_signal_L = np.array(self.yIn.reshape(len(self.yIn),1))
		stereo_signal_R = np.array(self.yIn.reshape(len(self.yIn),1))
		stereo_signal = np.hstack([stereo_signal_L,stereo_signal_R])
		m2s = stereo_track(stereo_signal, self.Fs, signal_type='stereo', filename = self.name[:-4]+'-stereo'+'.wav')
		return m2s


	###   效果器
	def eq(self,type,mode='auto',f0=440, Q=1., gain=0):
		if mode == 'auto':
			if type == 'lp':
				f0 = input('Input f0 (the cutoff frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				self.yIn, self.Fs = low_pass_filter(self.yIn,f0, Q, self.Fs)
			elif type == 'hp':
				f0 = input('Input f0 (the cutoff frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				self.yIn, self.Fs = high_pass_filter(self.yIn,f0, Q, self.Fs)
			elif type == 'ls':
				f0 = input('Input f0 (the frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				gain = input('How much gain do you want? :')
				self.yIn, self.Fs = low_shelf_filter(self.yIn,f0, gain, Q, self.Fs)
			elif type == 'hs':
				f0 = input('Input f0 (the frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				gain = input('How much gain do you want? :')
				self.yIn, self.Fs = low_shelf_filter(self.yIn,f0, gain, Q, self.Fs)
			elif type == 'peak':
				f0 = input('Input f0 (the center frequency):')
				Q = input('Input q value(which will decide the bandwidth of the filter):')
				self.yIn, self.Fs = peak_filter(self.yIn,f0, gain, Q, self.Fs)
			else:
				print('wrong filter type!')
		elif mode =='manul':
			if type == 'lp':
				self.yIn, self.Fs = low_pass_filter(self.yIn,f0, Q, self.Fs)
			elif type == 'hp':
				self.yIn, self.Fs = high_pass_filter(self.yIn,f0, Q, self.Fs)
			elif type == 'ls':
				self.yIn, self.Fs = low_shelf_filter(self.yIn,f0, gain, Q, self.Fs)
			elif type == 'hs':
				self.yIn, self.Fs = low_shelf_filter(self.yIn,f0, gain, Q, self.Fs)
			elif type == 'peak':
				self.yIn, self.Fs = peak_filter(self.yIn,f0, gain, Q, self.Fs)

		self.lufs = LUFS_detection(self.yIn,sr=self.Fs)
		self.rms = RMS_detection(self.yIn)
		self.peak = Peak_detection(self.yIn)

		return self.yIn, self.Fs, self.lufs , self.rms, self.peak

	def convolution(self,impluse):#直接传入一个mono_track的class类型
		if self.Fs == impluse.Fs:
			self.yIn = audio_effect(self.yIn, impluse.yIn, sr=self.Fs, seg_len = 512)
		else:
			print('wrong! different sample rate!')
		self.lufs = LUFS_detection(self.yIn,sr=self.Fs)
		self.rms = RMS_detection(self.yIn)
		self.peak = Peak_detection(self.yIn)
		return self.yIn, self.Fs, self.lufs , self.rms, self.peak

class stereo_track:	
	"""type for stereo signal input"""
	def __init__(self, yIn, Fs, signal_type, filename):
		self.name = str(filename)
		self.yIn = np.asanyarray(yIn)
		self.Fs = Fs
		self.signal_type = signal_type
		self.channel = 2
		self.left = np.array(self.yIn.T[0,:].reshape(len(self.yIn.T[0,:]),1)) # shape[xxx,1]
		self.right = np.array(self.yIn.T[1,:].reshape(len(self.yIn.T[1,:]),1)) # shape[xxx,1]
		self.lufs_L = LUFS_detection(self.left,sr=self.Fs)
		self.lufs_R = LUFS_detection(self.right,sr=self.Fs)
		self.rms_L = RMS_detection(self.left)
		self.rms_R = RMS_detection(self.right)
		self.peak_L = Peak_detection(self.left)
		self.peak_R = Peak_detection(self.right)

	###   基本功能
		# 调节电平
	def adjust_level(self,target):
		self.left = lufs_normalize(self.left,self.lufs_L,target_lufs=target)
		self.right = lufs_normalize(self.right,self.lufs_R,target_lufs=target)
		#renew parameters
		self.lufs_L = LUFS_detection(self.left,sr=self.Fs)
		self.lufs_R = LUFS_detection(self.right,sr=self.Fs)
		self.rms_L = RMS_detection(self.left)
		self.rms_R = RMS_detection(self.right)
		self.peak_L = Peak_detection(self.left)
		self.peak_R = Peak_detection(self.right)
		self.yIn = np.hstack([self.left,self.right])             #################  error
		return self.left, self.right, self.lufs_L, self.lufs_R, self.rms_L, self.rms_R, self.peak_L, self.peak_R , self.yIn

		#panning for stereo tracks (201)

	def pan(self,position):#-100 to 100
		if self.signal_type == 'mono':
			print('wrong! please first transform the signal into stereo.')

		elif self.signal_type == 'stereo':

			if position == 0:
				self.yIn = self.yIn

			elif position > 0 and position <100 :# pan right
				zero_pad = np.ones(len(self.left))
				zero_pad = np.array(zero_pad.reshape(len(self.left),1))
				target = abs((100-position)/100) #1-100
				zero_pad = zero_pad * target #1 x target
				self.left = zero_pad * self.left
				self.yIn = np.hstack([self.left,self.right])
				self.lufs_L = LUFS_detection(self.left,sr=self.Fs)
				self.rms_L = RMS_detection(self.left)
				self.peak_L = Peak_detection(self.left) 

			elif position < 0 and abs(position) < 100:
				zero_pad = zero_pad = np.ones(len(self.left))
				zero_pad = np.array(zero_pad.reshape(len(self.left),1))
				target = abs((100-abs(position))/100) #1-100
				zero_pad = zero_pad * target #1 x target
				self.right = zero_pad * self.right
				self.yIn = np.hstack([self.left,self.right])
				self.lufs_R = LUFS_detection(self.right,sr=self.Fs)
				self.rms_R = RMS_detection(self.right)
				self.peak_R = Peak_detection(self.right) 

			else:
				print('wrong! try value between -100 to 100 !')

	def change_name(self,new_name):
		self.name = str(new_name)
		return self.name

	def copy(self):
		copy_track = stereo_track(self.yIn, self.Fs, self.signal_type, self.name[:-4]+'-2'+'.wav')
		return copy_track

	def export(self,savepath):
		sf.write(savepath+'/'+self.name,self.yIn,self.Fs,'PCM_16')

	def export24(self,savepath):
		sf.write(savepath+'/'+self.name,self.yIn,self.Fs,'PCM_24')

	def stereo2mono(self, mode='mide'):
		if mode == 'mide':

			mono_signal = self.left + self.right
			max_value = max(np.max(abs(self.left)),np.max(abs(self.right)))
			mono_signal = MaxMinNormalization(mono_signal,i=max_value)

		elif mode == 'left':
			mono_signal = self.left

		elif mode == 'right':
			mono_signal = self.right

		s2m = mono_track(mono_signal, self.Fs, signal_type='mono',filename = self.name[:-4]+'-mono'+'.wav')
		return s2m

	###effect 效果器
	def eq(self,type,mode='auto',f0=440, Q=1., gain=0):

		left_channel = np.ravel(self.left) # matrix to vector
		right_channel = np.ravel(self.right)

		if mode == 'auto':
			if type == 'lp':
				f0 = input('Input f0 (the cutoff frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				left_channel, self.Fs = low_pass_filter(left_channel,f0, Q, self.Fs)
				right_channel, self.Fs = low_pass_filter(right_channel,f0, Q, self.Fs)
			elif type == 'hp':
				f0 = input('Input f0 (the cutoff frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				left_channel, self.Fs = high_pass_filter(left_channel,f0, Q, self.Fs)
				right_channel, self.Fs = high_pass_filter(right_channel,f0, Q, self.Fs)
			elif type == 'ls':
				f0 = input('Input f0 (the frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				gain = input('How much gain do you want? :')
				left_channel, self.Fs = low_shelf_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = low_shelf_filter(right_channel,f0, gain, Q, self.Fs)
			elif type == 'hs':
				f0 = input('Input f0 (the frequency):')
				Q = input('Input q value(which will decide the slope of the filter):')
				gain = input('How much gain do you want? :')
				left_channel, self.Fs = low_shelf_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = low_shelf_filter(right_channel,f0, gain, Q, self.Fs)
			elif type == 'peak':
				f0 = input('Input f0 (the center frequency):')
				Q = input('Input q value(which will decide the bandwidth of the filter):')
				left_channel, self.Fs = peak_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = peak_filter(right_channel,f0, gain, Q, self.Fs)
			else:
				print('wrong filter type!')

		elif mode == 'manul':
			if type == 'lp':
				left_channel, self.Fs = low_pass_filter(left_channel,f0, Q, self.Fs)
				right_channel, self.Fs = low_pass_filter(right_channel,f0, Q, self.Fs)
			elif type == 'hp':
				left_channel, self.Fs = high_pass_filter(left_channel,f0, Q, self.Fs)
				right_channel, self.Fs = high_pass_filter(right_channel,f0, Q, self.Fs)
			elif type == 'ls':
				left_channel, self.Fs = low_shelf_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = low_shelf_filter(right_channel,f0, gain, Q, self.Fs)
			elif type == 'hs':
				left_channel, self.Fs = low_shelf_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = low_shelf_filter(right_channel,f0, gain, Q, self.Fs)
			elif type == 'peak':
				left_channel, self.Fs = peak_filter(left_channel,f0, gain, Q, self.Fs)
				right_channel, self.Fs = peak_filter(right_channel,f0, gain, Q, self.Fs)

		self.left = np.array(left_channel.reshape(len(left_channel),1))
		self.right = np.array(right_channel.reshape(len(right_channel),1))
		self.yIn = np.hstack([self.left, self.right])
		self.lufs_L = LUFS_detection(self.left,sr=self.Fs)
		self.lufs_R = LUFS_detection(self.right,sr=self.Fs)
		self.rms_L = RMS_detection(self.left)
		self.rms_R = RMS_detection(self.right)
		self.peak_L = Peak_detection(self.left)
		self.peak_R = Peak_detection(self.right)

	def convolution(self,impluse):
		if self.Fs == impluse.Fs and impluse.signal_type =='mono':
			left_channel = np.ravel(self.left)
			right_channel = np.ravel(self.right)
			left_channel = audio_effect(left_channel, impluse.yIn, sr=self.Fs, seg_len = 512)
			right_channel = audio_effect(right_channel, impluse.yIn, sr=self.Fs, seg_len = 512)

			self.left = np.array(left_channel.reshape(len(left_channel),1))
			self.right = np.array(right_channel.reshape(len(right_channel),1))
			self.yIn = np.hstack([self.left, self.right])
			self.lufs_L = LUFS_detection(self.left,sr=self.Fs)
			self.lufs_R = LUFS_detection(self.right,sr=self.Fs)
			self.rms_L = RMS_detection(self.left)
			self.rms_R = RMS_detection(self.right)
			self.peak_L = Peak_detection(self.left)
			self.peak_R = Peak_detection(self.right)
		elif self.Fs != impluse.Fs:
			print('wrong! different sample rate!')
		elif mpluse.signal_type !='mono':
			print('impluse must be a mono impluse response signal!')

