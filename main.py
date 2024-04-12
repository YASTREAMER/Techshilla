from transformers import pipeline
import torch
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import time
import pickle 

from ModelFunction import * 
from const import *
from audio import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = pipeline(model="declare-lab/Flan-Alpaca-Large",device = device,token=access_token)


def main() -> None:

    print("Please enter the domain you want the questions to be in ")

    #Getting the domain they want the questions to be in
    RawText= input()

    InputText= "Tell me an interview question on " + RawText   
    
    #Generate a question for the user
    response = generate(generator,TokenLength, InputText)

    #Generate the answer to the question of the user
    responseAnswer = generate(generator, TokenLength, response["generated_text"])


    # Seeing whether or not the user is ready for the input or not
    Ready:int = 0
    while Ready ==0:

        print("Are you ready for the answer. Type 1 for yes and 0 for no")
        Ready= int(input())

        if  Ready == 1:
            time.sleep(1)
            print("Ready 3")
            time.sleep(1)
            print("2")
            time.sleep(1)
            print("1")
            time.sleep(1)

    record()
    bpm = tempo()

    Confident=pace(bpm)

    TranscribedAudio=SpeechToText()
    TranscribedAudio["text"]

    accuracy_answer=Sentence(TranscribedAudio, responseAnswer)
    
    print(f"The accuracy of your answer was {accuracy_answer*100}")

    feedback(TranscribedAudio,Confident,pace)

if __name__ =="__main__":
    main()
