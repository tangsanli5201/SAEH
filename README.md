# SAEH

TThis is the source code implemented by Keras for the paper "DEEP SUPERVISED AUTO-ENCODER HASHING FOR IMAGE RETRIEVAL" (in proceeding).

## Abstruct
Image hashing approaches map high dimensional images to compact binary codes since preserving similarities between image pairs. Although image label is the main information for supervised image hashing to generate hashing bits, such hashing bits should contain semantic information of various images. Therefore, we propose an effective supervised auto-encoder hashing method (SAEH) to generate a low dimensional binary codes in a point-wise manner of deep conventional neural network. The auto-encoder structure in SAEH is designed to simultaneously learn image features and generate hashing codes. Moreover, some extra relaxations for
generating binary hash codes are added to the objective function.  In our extensive experiments on several large scale image datasets, we validate that the auto-encoder structure can indeed increase the performance for supervised hashing and SAEH can achieve the best image retrieval results among other prominent supervised hashing methods.

## How to use this code?
### Training
1. install keras(>=0.8.0), tensorflow(>1.0) and numpy.
2. python hash_su_ae_train.py


### Testing
If you have already trained a hashing model, just edit the 'load_path' variable in the generate_hash.py to the model path AND run
python generate_hash.py.

## Experiments
### Ablation Study for Auto-encoder Structure
We remove the decoder structure from SAEH by setting the weight of decoder loss γ to 0 in the Eq.(7) in the paper, which is denoted as SAEH- in the paper (same in the following table). We also change the weight on the decoder loss γ to evaluate influences of the decoder network on the performance in image retrieval tasks.


###  Evaluation on SAEH and Other Methods
We compare our method with other hashing methods such as KSH，ITQ，DSH， CNNH+， SSDH(we implement SSDH by replacing the Alexnet structure to Resnet50 for fair comparasion).

