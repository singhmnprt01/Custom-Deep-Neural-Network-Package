# Custom DNN - Designed using NumPy only !
With customdnn the user can build his own neural network by simply passing few required/optional parameters. The user doesn't need to possess an advance knowledge of modern APIs like PyTorch, Tensorflow, Keras etc.

# Overview
The idea behind building this package is to learn and implement the mathematics behind neural network without using any single modern API, but matrices(NumPy) only. Below is the overview of the package:- 
- It is quite convenient to use the package as the user just needs to pass the Xs(feature dataset) and Y(target variable) in a dataframe and rest will be taken care.
- If the user wants change the default network architecture and build a customized NN, you can pass up to 9 parameters from learning rate to gradient descent algorithms (GDM,RMSprop,Adam).
- User can split the dataset by giving percentage of data to be test data, use it to predict and test his network's performance. (Area under the curve is current evaluation matrix)

# Using it
It is very handy and easy to use. User needs to install the package and give the dataset( x and y separately as 2 different dataframes ) along with percentage value (percent of data to be used as test data) to the split function in order to split the data into train and test
Now, training function can be called by passing training data along with user passed hyperparameters to build the customized Deep Network.
Output will give suitable parameters (weights and bias dictionary) and cost function vs epoch graph.
This parameter dictionary can be used to predict on test data and check model's performance as well (AUC value).

# Features of customdnn :-
  Set of user defined/chosen hyperparameters to make a custom DNN:-
  - Learning Rate : The rate of learning at which the gradient steps will be taken to minimise the cost
  - beta1 : Beta constant for Gradient Descent Momentum Optimisation Algorithm
  - beta2 : Beta constant for Root Mean Square prop Optimisation Algorithm
  - Mini Batch Size : To create customised mini-batches to amplify the processing and improve model accuracy/learning.
  - Network Size : A custom variable to design the number of layer of your network. It is exclusive of input and output layer
  - Gradient : Gradient Descent Optimisation algorithm selection field. The user can input any of the following three :-
        * GDM               - Gradient Descent Momentum
        * RMSprop           - Room Mean Square Prop
        * Adam              - Adaptive Momentum Estimation
  - Number of Epochs : Number of epochs/iterations for the network. 
  - Dropout size : To fuse some percentage of neurons
  - Neurons per layer 

  Other features to make customdnn more robust and vivid :-
  - Normalizing/Scaling Inputs
  - Xavier initialization of weights (fixed)
  - Model Evaluation using AUC (Area under the curve)
  - Cost function & cost function graph vs/ iterations 
  - Protection from exploding gradient descent (in most cases)
  
# Limitations
Currently the package is limited to binomial classification problems, and doesn't support/solve  multinomial , image and regression problems. 

# Upcoming Features
Batch-Norm
Early Stopping

# Website 
Other details of the package and the package itself can be found at https://github.com/singhmnprt01/Custom-Deep-Neural-Network/tree/master/customdnn

# Defaults
The current default output activation function is sigmoid and inner layers' is ReLU.

# References & credits
I got the inspiration from Andre NG's deeplearning.ai course & my mentor - Kiran R (Sr. Director AA&DS CoE,VMware). They inspired me to build my own custom deep neural network library.

# License
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

customdnn  Copyright (C) 2020  singhmnprt01@gmail.com
