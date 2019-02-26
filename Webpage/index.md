---
title:  'Home'
author: 
- name : Jeremy Fix
  affiliation : CentraleSupelec
keywords: [Deep learning, practicals]
...


**Forewords** These pages are written in markdown format and
generated with the [pandoc](http://www.pandoc.org). The stylesheet has been generated from [w3schools](https://www.w3schools.com/) and the codes are highlighted thanks to [pygment](http://pygments.org/). The transformation from pandoc markdown to html is done with a [Makefile](Makefile) and is using these [template files](templates).

## Installation

Below are some elements on how to install the required libraries to work on your own machine.

For the pytorch tutorials, you need [pytorch](https://pytorch.org/get-started/locally/), [tensorboardX](https://github.com/lanpa/tensorboardX), [matplotlib](https://matplotlib.org/), [tensorflow](https://tensorflow.org/install)

For the keras tutorial, you need [tensorflow](https://tensorflow.org/install), [keras](http://www.keras.io/#installation) and [matplotlib](https://matplotlib.org)

### CPU-Only, Ubuntu 18.04

On ubuntu 18.04, for the pytorch tutorials, it is as simple as :

``` console
mymachine:~:mylogin$ python3 -m pip install --user -U torch torchvision matplotplib tensorboardX tensorflow
```

On ubuntu 18.04, for the Keras tutorial, it is as simple as :

``` console
mymachine:~:mylogin$ python3 -m pip install --user -U tensorflow keras matplotplib
```




## Additional ressources

### Practicals/Lectures

- [Oxford’s CNN practicals](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/)
- [Nando de Freitas lectures and practicals](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
- [Stanford A. Karpathy practicals](http://cs231n.github.io/>)
- [Master Data Science Paris Saclay](https://github.com/m2dsupsdlclass/lectures-labs>)
- [Fast.ai, J. Howard](https://www.fast.ai/)

### Blog posts

- [An overview of gradient descent algorithms](http://ruder.io/optimizing-gradient-descent/)
- [Revisiting sequence to sequence learning, with focus on implementation details](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/)
- [The unreasonnable effectivness of recurrent neural networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Theories of Deep learning](https://stats385.github.io/)
