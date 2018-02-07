# Chatbot_with_Tensorflow
Chatbots are one of the most useful and real life use cases of AI.

A chatbot using Tensorflow 1.0.0. A Deep Natural Language Processing model named Seq2Seq is implemented.
Several stacked Recurrent Neural Networks (RNNs) are used to create this DNLP model.

***Please Note since my machine does not have significant processing power, it took several days (3 days) for the model to 
train.***

The code chatbot_tuned.py is heavily commented to help understand each step. 
The massive datasets used for creating the bot is also attached. 
The data set comprises of movie conversations between characters. 

The solution is broken down into several parts: 
1) Data Preprocessing
2) Building the Seq2Seq Model
3) Training the Seq2Seq Model
4) Testing the Seq2Seq Model

This chatbot can be used for your website or application. Enjoy! :) 
Feel free to play around with the hyperparameters to enjoy a conversation with the chatbot.

To reproduce this code: 
Make sure that Tensorflow 1.0.0 is installed
Python version 3.5 or above
You are inside an Anaconda environment and using Spyder

***A few extremely useful links which helped me along the way are shared below***
https://www.tensorflow.org/api_docs/python/tf/placeholder
https://www.tensorflow.org/api_docs/python/tf/fill
https://www.tensorflow.org/api_docs/python/tf/strided_slice
https://www.tensorflow.org/api_docs/python/tf/concat
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
https://www.tensorflow.org/programmers_guide/embedding
https://www.tensorflow.org/api_docs/python/tf/variable_scope
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/prepare_attention
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/attention_decoder_fn_train
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder
https://www.tensorflow.org/api_docs/python/tf/nn/dropout
https://www.tensorflow.org/programmers_guide/embedding
https://www.tensorflow.org/api_docs/python/tf/variable_scope
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/prepare_attention
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/attention_decoder_fn_inference
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder
https://www.tensorflow.org/api_docs/python/tf/nn/dropout
https://www.tensorflow.org/api_docs/python/tf/variable_scope
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
https://www.tensorflow.org/api_docs/python/tf/truncated_normal_initializer
https://www.tensorflow.org/api_docs/python/tf/zeros_initializer
https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence
https://www.tensorflow.org/api_docs/python/tf/Variable
https://www.tensorflow.org/api_docs/python/tf/random_uniform
https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
https://www.tensorflow.org/api_docs/python/tf/reset_default_graph
https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/placeholders#placeholder_with_default
https://www.tensorflow.org/api_docs/python/tf/shape
https://www.tensorflow.org/api_docs/python/tf/reverse 
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.reshape.html
https://www.tensorflow.org/api_docs/python/tf/name_scope
https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
https://www.tensorflow.org/api_docs/python/tf/ones
https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_clipping
https://www.tensorflow.org/api_docs/python/tf/clip_by_value
https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
https://pyformat.info/
https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
https://www.tensorflow.org/api_docs/python/tf/train/Saver


