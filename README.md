# test_js

This github repository is written by Mohammed Hamada Gasmallah (11mhg@queensu.ca) for the expressed purpose of testing the tensorflow js and serialization of both tensorflow graph and tensorflow layers models. There are discrepencies in both of these and a generalizable serialization library for tensorlfow models would be benificial. The ultimate purpose is to give developpers an idea of how to use DCP with tensorflow models of their choice for inference or training. 

# Graph vs Layers Models

Tensorflow graph models have been optimized for the expressed purpose of inferencing and cannot be trained on tensorflow js without playing around with some internal functions. Tensorflow layers models are quite bloated and slow for inferencing as they come with additional functionality that allow for training. 
Use whichever you please, but know the consequences of choosing one over the other.

One can typically convert one type of model to another using tensorflowjs\_converter (pip install tensorflowjs).

Pardon the programmer art of a website.
