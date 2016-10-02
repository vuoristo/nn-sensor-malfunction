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
Test accuracy: 0.263, active_splits: [True, False, False, False]
Test accuracy: 0.256, active_splits: [False, True, False, False]
Test accuracy: 0.282, active_splits: [False, False, True, False]
Test accuracy: 0.202, active_splits: [False, False, False, True]
Test accuracy: 0.504, active_splits: [False, False, True, True]
Test accuracy: 0.907, active_splits: [True, True, False, False]
Test accuracy: 0.796, active_splits: [False, True, True, False]
Test accuracy: 0.982, active_splits: [True, True, True, False]
Test accuracy: 0.995, active_splits: [True, True, True, True]

Testing full_model trained with partial training data after 20000 steps
Test accuracy: 0.731, active_splits: [True, False, False, False]
Test accuracy: 0.792, active_splits: [False, True, False, False]
Test accuracy: 0.846, active_splits: [False, False, True, False]
Test accuracy: 0.737, active_splits: [False, False, False, True]
Test accuracy: 0.912, active_splits: [False, False, True, True]
Test accuracy: 0.957, active_splits: [True, True, False, False]
Test accuracy: 0.952, active_splits: [False, True, True, False]
Test accuracy: 0.976, active_splits: [True, True, True, False]
Test accuracy: 0.977, active_splits: [True, True, True, True]

Testing fixed_model trained with partial training data after 20000 steps
Test accuracy: 0.734, active_splits: [True, False, False, False]
Test accuracy: 0.782, active_splits: [False, True, False, False]
Test accuracy: 0.828, active_splits: [False, False, True, False]
Test accuracy: 0.689, active_splits: [False, False, False, True]
Test accuracy: 0.908, active_splits: [False, False, True, True]
Test accuracy: 0.936, active_splits: [True, True, False, False]
Test accuracy: 0.947, active_splits: [False, True, True, False]
Test accuracy: 0.971, active_splits: [True, True, True, False]
Test accuracy: 0.981, active_splits: [True, True, True, True]
```

Conclusion
----------
In the initial experiments the performance of the classifier in classifying partially obstructed inputs does not appear to improve with fixing the softmax layer.
