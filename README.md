# EfficientNet-B0-with-bi-GRU-on-medical-image
Introduce a new scaling neural network model that carefully balancing network depth, width, and resolution can lead to better performance

Firstly, a new model scaling network Efficientnet (a kind of CNN) is used as the backbone network to extract colposcopy image spatial features. 

And then, feature fusion was realized through concat the features extracted from EfficientNet and 1x1 convolution, these layers rearrange and combine the connected features to form new features. 

Finally the concated long sequence features were sent to GRU to realize integrated feature extraction, and the integrated features were sent to the classification layer to realize classification algorithm colposcopy image of in this paper.

## How to run
To run this program, simply type the following command:

`python train_classifier.py`

## Package dependency
1. scikit-lean==0.21.2
2. pytorch==1.0.1
3. torchvision==0.2.1
4. pillow==5.4.1
5. efficientnet-pytorch==0.7.0

In my practice, this implementation needs to be above pytorch 1.0 to work. If there is an incompatibility problem, please feel free to open a problem.
