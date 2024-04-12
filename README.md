# Techshilla

This is the offical submission for the techshilla's problem statement by Ravindra Bhawan's team. 

## Models used

For the problem statement a number of models had to be used and were fine tuned accordingly to fit our need. These can be divided into 4 sections. These include 

1) [Question generation.](#question-generation)
2) [Speech to text convertor.](#speech-to-text-generation)
3) [Grammer error detection and correction.]()
4) [Feedback.]()

These models were fit into a certain pipeline to be used effectively. A detailed section for each of these could be found below.

# Question Generation

For generating the question we used Flan-Alpaca-Large / Llama 2 which are open source models but for them to be used we had to fine tune them. This was done by using a custom dataset which was made for scratch. The dataset contained the type of question to be generated and then the question to be generated

The fine tuning was a major huddle as using a untrained model would result in the question being used to not have a depth to them. 

The link for the custom dataset is [here](https://docs.google.com/spreadsheets/d/1K8H9LTCcvwUZwM5rkI62FLiknmnl_as5Kw0jyJdGBBI/edit?usp=sharinghttps://docs.google.com/spreadsheets/d/1K8H9LTCcvwUZwM5rkI62FLiknmnl_as5Kw0jyJdGBBI/edit?usp=sharing)

# Speech to Text Generation

This was one of the biggest task that had to be handled. The model not only had to convert speech to text but it also had to be efficient enough for it to not cause a bottle neck. Thus we decided to use OpenAI's whisper model. With this model we were able to convert speech to text.

When answering an interview question, one major factor to be considered was that whether or not the speaker was speaking at an optimatl pace. Usually the optimal pace depends on the type of question asked and the person speaking but, generally it should lie between 120 to 150 beats per minute. [Librosa](https://github.com/librosa/librosa) was used to find the Beats per minute/ Tempo and the user was accordingly given a review.