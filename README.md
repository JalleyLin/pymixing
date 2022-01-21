# Pymixing

## 项目描述 Project description

*Pymixing*是一个基于*soundfile*，*numpy*，*librosa*, *scipy* 和 *pyloudnorm* 的python音频处理库。启发于传统调音台设备和数字音频工作站，旨在通过简洁的代码，为音频领域的开发者、研究者提供帮助。目前，pymixing不仅支持读写wav格式的单声道文件，还支持相同格式的立体声文件，并且已经可以根据需求导出相应的48k & 24bit或44.1k & 16bit的wav格式的音频文件。

*Pymixing* is a python audio processing / analysis library based on *soundfile*, *numpy*, *librosa*, *scipy* and *pyloudnorm*.Inspired by traditional analog console and digital audio workstations, it aims to provide audio researchers with flexible assistance to deal with the complex audio processing / detection tasks. Currently, pymixing can not only supports the mono files in wav format, but also the stereo files. Besides,it is already possible to export the corresponding 48k / 24bit or 44.1k / 16bit wav format mono/stereo audio files as required.

## 下载 Installation

Pymixing已经发布在[PYPI](https://pypi.org/project/pymixing/)上，在终端（Mac电脑）中输入以下代码就可以实现下载。

Pymixing has been released on [PYPI](https://pypi.org/project/pymixing/)，In a modern Python, you can use the code below to download and install the latest release of Pymixing and its dependencies. 

~~~python
pip install pymixing
~~~

## 功能介绍 Functions

那么pymixing具有什么功能呢？

So what can we do with pymixing？

### 音频导入/导出  Load/Export Functions

**读取wav格式的文件**，可以调用`load_track(path, filename)`函数实现。

If you need to **open a wav format mono & stereo aduio file**，you can try `load_track(path, filename)`.

~~~python
import pymixing as pymx
track1 = pymx.load_track('F:/products/J_Full','09_ElecGtr2.wav') #'.wav' is needed in filename.
~~~

将track1**导出**为16bit的wav文件，可以调用`export(self,savepath)`函数实现。

If you need to **export track1 as a 16 bit wav format mono & stereo aduio file**，you can try `export(self,savepath)`.

~~~python
track1.export('F:/products/J_Full') #the saved filename is as same as the orignal load file.
~~~

保存的文件名是默认与读取文件名一致的，如果想要**改名**后再导出这个文件，可以使用 `change_name(self,new_name)`
The saved filename is as same as the orignal load file，if you want to **change the name of the wav file** to be saved,please check `change_name(self,new_name)`.

~~~python
track1.change_name('demo.wav')
track1.export('F:/products/J_Full') #the saved filename is 'demo.wav' now.
~~~

此外，如果希望**导出24bit的音频文件**，可以尝试`export24(self,savepath)`

Besides，If you need to **export track1 as a 24 bit wav format mono & stereo aduio file**，you can try `export(self,savepath)`.

~~~python
track1.export24('F:/products/J_Full') 
~~~


### 检测音频参数  Checking audio features

和调音台 & daw类似，pymixing也可以**检测多种音频参数**，并且通过简单的函数轻松调用查看，例如，我们希望查看之前读取的track1音轨的峰值电平（peak）和lufs电平参数,可以直接用print函数打印出来查看。

Like many DAWs, pymixing can also be used to **check many different parameters** like peak or LUFS level.

~~~python
print(track1.peak) #peak of the audio signal (dBFS)
print(track1.lufs) # LUFS level (based on ITU-R BS.1770-4) of the audio signal
print(track1.rms)  # RMS level of the audio signal (dBFS)
~~~

### 音频剪辑 Audio Editing

与[Pydub](http://pydub.com/)类似，Pymixing还内置有强大的音频剪辑功能。

Like [pydub](http://pydub.com/)， pymixing also has powerful eaditing functions.

调用`cut(audio, time, crossfade=True, cross_time=200)`，它能进行高精度的（精度为0.001s，但是没有按照60s换算）**音频剪切**。例如，将track1在第15.2秒处进行切分，切分为两个部分，暂且命名为a和b。

Using `cut(audio, time, crossfade=True, cross_time=200)`, it can **slice audio with（or without）the crossfade**. For example，cut track1 into 2 parts, and the former part has a length of 15.2 seconds.

~~~python
a, b = cut(track1, time=15.2)
~~~

调用`adjust_level(self,target)`,可以根据lufs电平**调节响度**，例如，将切分后的track1的前一部分（a）的lufs电平调节为-22lufs.

Using`adjust_level(self,target)`, it can **adujust lufs level of the audio clip**. For example，adusting the lufs level of the former part of track1 to -22LUFS.

~~~python
a.adjust_level(traget=-22)
~~~

尝试调用`splice(a, b, crossfade=True, cross_time=200)`，它能将切分的音频片段重新**拼接**起来。

Using `splice(a, b, crossfade=True, cross_time=200)`, it can **concatenate 2 audio clips**. 

~~~python
track2 = splice(a, b)
~~~

如果想要**复制音频片段**，可以用`copy(self)`。

If you want to **copy the audio clip**, try `copy(self)`.

~~~python
track3 = track2.copy()
~~~

如果想要**合并两条音轨（两个音频文件）**，可以用`group(a,b,name)`，使用时请务必保持两条音轨前段对齐。

If you want to **bounce 2 tracks into 1 track**, try `group(a,b,name)`.Before using, please keep 2 tracks aligned.

~~~python
track_group = group(track2, track3, name='group_file.wav')
~~~

最后，可以用`normalize(self,i=0.5)`**标准化（归一化）音频**，i代表着振幅的最大值（取值范围为0-1）。

Finally ,you can use `normalize(self,i=0.5)` to **normalize the track**, i means the peak (0 to 1) of the track.

~~~python
track_group.normalize(i=1)
~~~

### 音频效果 Audio Effects

Pymixing还内置有常见的音频效果。

The common audio effects are also provided by pymixing.

如果想使用**均衡器（EQ**），可以调用`eq(self,type,mode='auto',f0=440, Q=1., gain=0)`。pymixing内置的均衡器（EQ）提供有两种模式，分别是*auto*(自动)和*manul*（手动），自动模式通过交互的方式让使用者提供所需的参数，而手动模式则在eq函数中直接输入所需的参数。

If you want to use the **equalizer(EQ)** to process the audio signal, please try `eq(self,type,mode='auto',f0=440, Q=1., gain=0)`. 2 modes (*auto* and *manul*) have been provided, in *auto* mode you will need to type in the parameters as required, while in *manul* mode you will need to type in the parameters as usual.

Pymixing提供五种滤波器类型供选择，分别为低通滤波器（'lp'）、高通滤波器（'hp'）、低搁架式滤波器（'ls'）、高搁架式滤波器（'hs'）、钟型滤波器（'peak'）。函数所需的输入参数如下所示。

Pymixing provides 5 filter types: low pass fileter ('lp'), high pass filter ('hp'), low shelf filter ('ls'), high shelf filter ('hs') and peak filter ('peak'). The parameters needed is showing below.

~~~shell
type:    滤波器类型    filter types                                          'lp'/'hp'/'ls'/'hs'/'peak'
mode     均衡器模式    mode of eq                                            'auto'/'manul'
f0:      中心频率      center frequency point                                0 to (samplerate/2)     
Q:       峰值带宽      Q value (bandwidth)                                   0 to 10 (dBFS)
gain:    增益          How much you want to boost (+) or substract (-)       0 to 20 (dBFS)
~~~


例如，如果想用钟型滤波器增加'track_group'在500Hz处的能量，增量为3dB，代码如下:

For example, if you want the boost 'track_group' auroud 500Hz with 3dB:

~~~python
track_group.eq(type='peak',mode='manul',f0=500,Q=1.,gain=3)
~~~


