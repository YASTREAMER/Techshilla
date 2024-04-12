# Techshilla

This is the offical submission for the techshilla's problem statement by Ravindra Bhawan's team. 

## Models used

For the problem statement a number of models had to be used and were fine tuned accordingly to fit our need. These can be divided into 4 sections. These include 

1) [Question generation.](#question-generation)
2) [Speech to text convertor.]()
3) [Grammer error detection and correction.]()
4) [Feedback.]()

These models were fit into a certain pipeline to be used effectively. A detailed section for each of these could be found below.

# Question Generation

For generating the question we used Llama 2 which is open source model but for it to be used we had to fine tune it. This is done by using a custom dataset which was made for scratch. The dataset contained the type of question to be generated and then the question to be generated

The fine tuning was a major huddle as using a untrained model would result in the question being used to not have a depth to them. 