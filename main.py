from transformers import pipeline
import torch
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import whisper

from ModelFunction import * 
from const import *
from audio import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = pipeline(model="declare-lab/Flan-Alpaca-Large",device = device,token=access_token)

reponse=generate(generator,TokenLength)

record()
bpm = tempo()

boolean=pace(bpm)
print(boolean)

print(reponse)