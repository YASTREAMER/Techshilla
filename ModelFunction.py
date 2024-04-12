


def generate(model,maxlength) -> str:

  print("Hello")
  inputText= input()
  GeneratedText=model(inputText,max_length=16192, do_sample=False)

  return(GeneratedText)
