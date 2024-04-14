from transformers.utils.import_utils import is_nltk_available
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf

device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError ('GPU device not found')



def generate(model,maxlength,inputText) -> str:

  
  GeneratedText=model(inputText,max_length=16192, do_sample=False)

  return(GeneratedText)

def Sentence(Sentense1, Sentense2) -> float:

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

  #Compute embedding for both lists
  embedding_1= model.encode(Sentense1, convert_to_tensor=True)
  embedding_2 = model.encode(Sentense2, convert_to_tensor=True)

  similar=util.pytorch_cos_sim(embedding_1, embedding_2)

  return similar


def feedback(response,responsean,Confident,pace) -> None:

  print(f"Your answer was {response}, while it should have been {responsean}")
  print(f"{response}.")

  if Confident:
    print(f"You were quite confident the your average pace was around:- {pace}")

  else:
    print(f"Your pace was quite off it was around:- {pace}. Your average pace should be around 128 and the range should be around 120 to 150")


def Grammer(model) -> int:

  #Predicting whether the sentence has grammatical error

  return