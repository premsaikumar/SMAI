SIMPLE_2d
To reconstruct the toy dataset run all the cells in the simple_2d.ipynb

MNIST and CIFAR
To perform training and reconstructions, the program "Main.py" needs to be executed with the appropriate parameters. The directory "command_line_args" contains command-line instructions with the necessary arguments to replicate the training of the models provided, as well as their corresponding reconstructions (which are analyzed in the notebooks).


Training
To train the MLP models:

 - CIFAR10 model (for reproduction run ```command_line_args/train_cifar10_vehicles_animals.txt```)

The check points for the trained model are located at
https://github.com/SrikantKonduri/SMAI-Project/blob/defcdb53340e7b099fc8c01a420705420e576d3f/49000_x.pth


 - MNIST model (for reproduction run ```command_line_args/train_mnist_odd_even_args.txt```)


Reconstructions



These reconstructions can be reproduced by running the following commandlines:

- CIFAR10: ```command_line_args/reconstruct_cifar10_b9dfyspx_args.txt``` and ```command_line_args/reconstruct_cifar10_k60fvjdy_args.txt```

The check points for the reconstructions are located at
https://github.com/SrikantKonduri/SMAI-Project/blob/defcdb53340e7b099fc8c01a420705420e576d3f/49000_x.pth

- MNIST: ```command_line_args/reconstruct_mnist_kcf9bhbi_args.txt``` and ```command_line_args/reconstruct_mnist_rbijxft7_args.txt```
The check points for the reconstructions are located at
https://github.com/SrikantKonduri/SMAI-Project/blob/8699d427d6070b703a1966e48cafa9343c0a85a2/mni.pth


