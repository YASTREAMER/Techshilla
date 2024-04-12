from transformers.utils.import_utils import is_nltk_available
from sentence_transformers import SentenceTransformer, util

def generate(model,maxlength) -> str:

  print("Hello")
  inputText= input()
  GeneratedText=model(inputText,max_length=16192, do_sample=False)

  return(GeneratedText)

def Sentence(Sentense1, Sentense2) -> float:

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

  #Compute embedding for both lists
  embedding_1= model.encode(Sentense1, convert_to_tensor=True)
  embedding_2 = model.encode(Sentense2, convert_to_tensor=True)

  similar=util.pytorch_cos_sim(embedding_1, embedding_2)

  return similar


def feedback(response,Confident,pace) -> None:


  print(f"{response}.")

  if Confident:
    print(f"You were quite confident the your average pace was around:- {pace}")