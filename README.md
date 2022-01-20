# Pymixing

## 项目描述 Project description

*Pymixing*是一个基于*soundfile*，*numpy*，*librosa*, *scipy* 和 *pyloudnorm* 的python音频处理库。启发于传统调音台设备和数字音频工作站，旨在通过简洁的代码，为音频领域的开发者、研究者提供帮助。目前，pymixing不仅支持读写wav格式的单声道文件，还支持相同格式的立体声文件，并且已经可以根据需求导出相应的48k & 24bit或44.1k & 16bit的wav格式单声道文件。

*Pymixing* is a python audio processing / analysis library based on *soundfile*, *numpy*, *librosa*, *scipy* and *pyloudnorm*.Inspired by traditional analog console and digital audio workstations, it aims to provide audio researchers with flexible assistance to deal with the complex audio processing / detection tasks. Currently, pymixing can not only supports the mono files in wav format, but also the stereo files. Besides,it is already possible to export the corresponding 48k / 24bit or 44.1k / 16bit wav format mono/stereo audio files as required.

## 下载 Installation

Pymixing已经发布在PYPI上，在终端（Mac电脑）中输入以下代码就可以实现下载。

Pymixing has been released on PYPI，In a modern Python, you can use the code  below to download and install the latest release of Pymixing and its dependencies. 

~~~python
pip install pymixing
~~~

## 功能介绍 Functions

那么pymixing具有什么功能呢？

So what can we do with pymixing？

### 音频导入/导出  Load/Export Functions

读取wav格式的文件，可以调用`load_track(path, filename)`函数实现。

If you need to open a wav format mono & stereo aduio file，you can try `load_track(path, filename)`.

~~~python
import pymixing as pymx
track1 = pymx.load_track('F:/products/J_Full','09_ElecGtr2.wav') #'.wav' is needed in filename.
~~~

将track1导出为16bit的wav文件，可以调用`export(self,savepath)`函数实现。

If you need to export track1 as a 16 bit wav format mono & stereo aduio file，you can try `export(self,savepath)`.

~~~python
track1.export('F:/products/J_Full') #the saved filename is as same as the orignal load file.
~~~

保存的文件名是默认与读取文件名一致的，如果想要改名后再导出这个文件，可以使用 `change_name(self,new_name)`
The saved filename is as same as the orignal load file，if you want to change the name of the wav file to be saved,please check `change_name(self,new_name)`.

~~~python
track1.change_name('demo.wav')
track1.export('F:/products/J_Full') #the saved filename is 'demo.wav' now.
~~~

此外，如果希望导出24bit的音频文件，可以尝试`export24(self,savepath)`

Besides，If you need to export track1 as a 24 bit wav format mono & stereo aduio file，you can try `export(self,savepath)`.

~~~python
track1.export24('F:/products/J_Full') 
~~~


### 检测音频参数  Checking audio features

和调音台 & daw类似，pymixing也可以检测多种音频参数，并且通过简单的函数轻松调用查看，例如，我们希望查看之前读取的track1音轨的峰值电平（peak）和lufs电平参数,可以直接用print函数打印出来查看。

Like many DAWs, pymixing can also be used to check many different parameters like peak or LUFS level.

~~~python
print(track1.peak) #peak of the audio signal (dBFS)
print(track1.lufs) # LUFS level (based on ITU-R BS.1770-4) of the audio signal
print(track1.rms)  # RMS level of the audio signal (dBFS)
~~~

### 音频剪辑 Audio Editing

Pymixing还内置有强大的音频剪辑功能。

Pymixing also has powerful eaditing functions.

调用`cut(audio, time, crossfade=True, cross_time=200)`，它能进行高精度的（精度为0.001s，但是没有按照60s换算）音频剪辑。例如，将track1在第15.2秒处进行切分，切分为两个部分，暂且命名为a和b。

Using `cut(audio, time, crossfade=True, cross_time=200)`, it can slice audio with（or without）the crossfade. For example，cut track1 into 2 parts, and the former part has a length of 15.2 seconds.

~~~python
a, b = cut(track1, time=15.2)
~~~

调用`cut(audio, time, crossfade=True, cross_time=200)`

