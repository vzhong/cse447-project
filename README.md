# CSE 447 Project
Group Shreyjun

# Instructions
run `bash src/predict.sh <path_to_test_data> <path_to_predictions>`

# Answers to checkpoint 1

TODO: Make these better

## Dataset we're using: 
For now we're using a subset of the One Billion Word Language Modeling Benchmark

## Method:
For now, we're using a smoothing between a unigram, bigram, and trigram model. This gives us a 61.538% accuracy which is a pretty good starting point. Additionally, thanks to heavy pre-processing, predicitions are made almost instantly. This makes such a model an excellent choice from processing time perspective. That said, we might replace this with a neural net for better accuracy. 