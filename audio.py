# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import librosa
import whisper

from const import *


def record() -> None:

    recording = sd.rec(int(duration * freq),
                    samplerate = freq, channels = 2)

    sd.wait()

    write("Input/input.wav", freq, recording)


def tempo() -> float:

    audio_file = librosa.load('Input/input.wav')
    y, sr = audio_file

    #Tempo Detection 
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)    

    return tempo

def SpeechToText() -> str:

    model = whisper.load_model("base")

    result = model.transcribe("Input/input.wav")
    
    return result


def pace(bpm) -> bool:

    if bpm > 150 and bpm < 120:
        
        return False
    
    else:
        return True