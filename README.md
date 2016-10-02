nn-sensor-malfunction
=====================

Requirements
------------
* pip
* virtualenv (recommended)
* python 3
* numpy
* tensorflow

Installation
------------
* Clone this repository
* Create new virtualenv environment
* For GPU enabled version of tensorflow, check installation instructions on the tensorflow [site](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
* Install requirements with `pip install -r requirements.txt`

Running the experiments
-----------------------
* Run the experiments with `python mnist_malfunction.py`

Description
-----------
Making deep networks more robust to sensor malfunction

nn-sensor-malfunction is an experiment on combining sensor outputs in a neural network in a way that is more robust to sensor malfunction than full end-to-end training.

Imagine an automation system that has multiple sensors with enough redundancy to operate in principle, even if any one of the sensors is not working. Feedforward neural networks are in trouble when part of the input is missing. A possible solution would be to have a separate network for each sensor, only combining the signals just before the softmax nonlinearity at the output (assuming a classification task). One could first train only the bias at the softmax layer and then each network separately in combination with the fixed bias. This would closely correspond to the naive Bayes classifier that treats inputs independently.

In the experiment the task is to train a network to recognize MNIST digits with parts of the input obstructed. The network consists of four convolutional neural networks that process one quarter of the input image. First, the full network is trained with full training images. Next, the full network is trained with partial inputs. Last, the softmax layer of the network is fixed and the network is trained with partial inputs. The network performance is tested with full and partial inputs between each training step. The second training step is a control step to check if fixing the softmax layer gives any benefit.

Results
-------
In the following the results of running the experiment with default parameters is represented. The `active_splits` correspond to the quarters of the input enabled.

`full_model` has a trainable softmax layer.
In the `fixed_model` the softmax layer weights are copied from the initial model and then made untrainable.

```
Testing full_model after 20000 steps
Test accuracy: 0.314, active_splits: [True, False, False, False]
Test accuracy: 0.247, active_splits: [False, True, False, False]
Test accuracy: 0.233, active_splits: [False, False, True, False]
Test accuracy: 0.133, active_splits: [False, False, False, True]
Test accuracy: 0.435, active_splits: [False, False, True, True]
Test accuracy: 0.910, active_splits: [True, True, False, False]
Test accuracy: 0.841, active_splits: [False, True, True, False]
Test accuracy: 0.971, active_splits: [True, True, True, False]
Test accuracy: 0.996, active_splits: [True, True, True, True]

Testing full_model trained with partial training data after 20000 steps
Test accuracy: 0.730, active_splits: [True, False, False, False]
Test accuracy: 0.791, active_splits: [False, True, False, False]
Test accuracy: 0.829, active_splits: [False, False, True, False]
Test accuracy: 0.735, active_splits: [False, False, False, True]
Test accuracy: 0.921, active_splits: [False, False, True, True]
Test accuracy: 0.953, active_splits: [True, True, False, False]
Test accuracy: 0.967, active_splits: [False, True, True, False]
Test accuracy: 0.980, active_splits: [True, True, True, False]
Test accuracy: 0.981, active_splits: [True, True, True, True]

Testing fixed_model trained with partial training data after 20000 steps
Test accuracy: 0.734, active_splits: [True, False, False, False]
Test accuracy: 0.768, active_splits: [False, True, False, False]
Test accuracy: 0.814, active_splits: [False, False, True, False]
Test accuracy: 0.698, active_splits: [False, False, False, True]
Test accuracy: 0.901, active_splits: [False, False, True, True]
Test accuracy: 0.922, active_splits: [True, True, False, False]
Test accuracy: 0.939, active_splits: [False, True, True, False]
Test accuracy: 0.964, active_splits: [True, True, True, False]
Test accuracy: 0.976, active_splits: [True, True, True, True]
```

Conclusion
----------
In the initial experiments the performance of the classifier in classifying partially obstructed inputs does not appear to improve with fixing the softmax layer.
