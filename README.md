# code2vec
This is a python implementation of [code2vec](https://arxiv.org/pdf/1803.09473.pdf) that uses TensorFlow2.1 

# Module Architecture
- **[repo folder]/configs**: contains the configuration file tunnable in this repo.
- **[repo folder]/cores**: contains the model definitions for calculating the loss and inference.
- **[repo folder]/utils**: contains the trainer, batch_generator, data-preprocessers.
- **[repo folder]/scripts**: contains the script for triggering the functionalities of this repo. 


# To Get Started

## Set up the virtual environment using Anaconda.
We recommend our potential users to use [Anaconda](https://www.anaconda.com/) as the primary virtual environment. 

```shell
$conda create --name [your environ name] python=3.6.8
$conda activate [your environ name]
$pip install -r requirements.txt
```	
