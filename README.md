# MGAN_Paper
is a novel deep learning framework designed for malicious traffic detection in network security. It combines Graph Attention Networks (GATs) for capturing spatial dependencies and Transformer-based sequence modeling for identifying temporal attack behaviors. 
# Installation instruction
pip install -r requirements.txt

#Implementation
Here we provide the implementation of a MGAN layer in TensorFlow, along with a minimal execution example (on the CICIDS2017 dataset). The repository is organised as follows:

dataset/ contains the necessary dataset files for CICIDS2017;
models/models.py contains the implementation of the MGAN(Model);
layers.py contains the implementation of the MultiGraphConvolution(Layer);
Finally, train.py puts all of the above together and may be used to execute a full training run on a CICIDS2017.
