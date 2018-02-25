# SAEH

This is the source code implemented by Keras for the paper "DEEP SUPERVISED AUTO-ENCODER HASHING FOR IMAGE RETRIEVAL" (in proceeding).

## Abstruct
Image hashing approaches map high dimensional images to compact binary codes since preserving similarities between image pairs. Although image label is the main information for supervised image hashing to generate hashing bits, such hashing bits should contain semantic information of various images. Therefore, we propose an effective supervised auto-encoder hashing method (SAEH) to generate a low dimensional binary codes in a point-wise manner of deep conventional neural network. The auto-encoder structure in SAEH is designed to simultaneously learn image features and generate hashing codes. Moreover, some extra relaxations for
generating binary hash codes are added to the objective function.  In our extensive experiments on several large scale image datasets, we validate that the auto-encoder structure can indeed increase the performance for supervised hashing and SAEH can achieve the best image retrieval results among other prominent supervised hashing methods.

## How to use this code?
1. install keras and tensorflow
2. python train_supervised
