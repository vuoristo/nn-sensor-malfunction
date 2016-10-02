from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np

def weight_variable(shape, trainable=True):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable=trainable)

def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                        padding='SAME')

def build_net(x, keep_prob):
  W_conv1 = weight_variable([2,2,1,32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,14,14,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([2,2,32,64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([4 * 4 * 64, 512])
  b_fc1 = bias_variable([512])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([512, 10])
  b_fc2 = bias_variable([10])

  trainable_vars = {
    'W_conv1': W_conv1,
    'b_conv1': b_conv1,
    'W_conv2': W_conv2,
    'b_conv2': b_conv2,
    'W_fc1': W_fc1,
    'b_fc1': b_fc1,
    'W_fc2': W_fc2,
    'b_fc2': b_fc2,
  }
  return tf.matmul(h_fc1_drop, W_fc2) + b_fc2, trainable_vars

NUM_NETS = 4
INPUT_SIZE = 784 / NUM_NETS
def build_model(trainable_last_layer=True):
  # build NUM_NETS conv networks
  inputs = [tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
            for _ in range(NUM_NETS)]
  keep_prob = tf.placeholder(tf.float32)
  net_outputs = []
  net_trainable_vars = []
  for x in inputs:
    outputs, ntv = build_net(x, keep_prob)
    net_outputs.append(outputs)
    net_trainable_vars.append(ntv)

  # build softmax combiner
  softmax_inputs = tf.concat(1, net_outputs)
  W_fc_full = weight_variable([NUM_NETS*10,10],
                              trainable=trainable_last_layer)
  b_fc_full = bias_variable([10], trainable=trainable_last_layer)
  y_conv = tf.nn.softmax(tf.matmul(softmax_inputs, W_fc_full) + b_fc_full)

  # build loss, training and accuracy operations for the model
  targets = tf.placeholder(tf.float32, shape=[None, 10])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(y_conv),
                                 reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(targets,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # return placeholders and callable operators
  model = {
    'inputs': inputs,
    'keep_prob': keep_prob,
    'net_outputs': net_outputs,
    'softmax_inputs': softmax_inputs,
    'softmax_W': W_fc_full,
    'softmax_b': b_fc_full,
    'targets': targets,
    'train_step': train_step,
    'accuracy': accuracy,
    'trainable_vars': net_trainable_vars,
  }

  return model

TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 1000
def get_batch(full_batch, train, active_splits=None):
  if train:
    batch = mnist.train.next_batch(TRAIN_BATCH_SIZE)
  else:
    batch = mnist.test.next_batch(TEST_BATCH_SIZE)

  square_batch = np.reshape(batch[0], (-1,28,28))
  v_split = 28 / int(np.sqrt(NUM_NETS))
  h_split = 28 / int(np.sqrt(NUM_NETS))
  batch_splits = []
  if full_batch:
    num_splits_kept = NUM_NETS
  else:
    num_splits_kept = np.random.randint(1,NUM_NETS)

  if active_splits == None:
    active_splits = num_splits_kept*[True] + (
        NUM_NETS-num_splits_kept)*[False]
    np.random.shuffle(active_splits)

  for i, split_active in enumerate(active_splits):
    x = int(i % np.sqrt(NUM_NETS))
    y = int(i / np.sqrt(NUM_NETS))
    split = square_batch[:, x*h_split:(x+1)*h_split,
        y*v_split:(y+1)*v_split]
    if not split_active:
      split = np.zeros_like(split)
    batch_splits.append(np.reshape(split, (-1, INPUT_SIZE)))

  return batch_splits, batch[1], active_splits

def update_trainable_variables(active_splits, net_index, model):
  variables_to_train = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
  if active_splits[net_index]:
    for _, trainable_var in model['trainable_vars'][net_index].items():
      if trainable_var not in variables_to_train:
        variables_to_train.append(trainable_var)
  else:
    for _, trainable_var in model['trainable_vars'][net_index].items():
      if trainable_var in variables_to_train:
        variables_to_train.remove(trainable_var)

DROPOUT_PROB = 0.5
def run_training(session, model, num_steps, full_depth):
  for step in range(num_steps):
    splits, labels, active_splits = get_batch(full_depth, True)
    feed_dict={}
    for idx, input_x in enumerate(model['inputs']):
      feed_dict[input_x] = splits[idx]
      update_trainable_variables(active_splits, idx, model)

    feed_dict[model['targets']] = labels
    feed_dict[model['keep_prob']] = DROPOUT_PROB
    session.run(model['train_step'], feed_dict=feed_dict)
    if step % 100 == 0:
      feed_dict[model['keep_prob']] = 1.0
      acc = session.run(model['accuracy'], feed_dict=feed_dict)
      print('step: {}, accuracy: {}, active_splits: {}'.format(
        step, acc, active_splits))

def run_test(session, model, full_depth, active_splits=None):
  splits, labels, active_splits = get_batch(full_depth, False, active_splits)
  feed_dict={}
  for idx, input_x in enumerate(model['inputs']):
    feed_dict[input_x] = splits[idx]
  feed_dict[model['targets']] = labels
  feed_dict[model['keep_prob']] = 1.0
  acc = session.run(model['accuracy'], feed_dict=feed_dict)
  print('Test accuracy: {}, full_depth: {} active_splits: {}'.format(
      acc, full_depth, active_splits))

def get_fixed_updates(full_model, fixed_model):
  updates = []
  updates.append(fixed_model['softmax_W'].assign(full_model['softmax_W']))
  updates.append(fixed_model['softmax_b'].assign(full_model['softmax_b']))
  return updates

FULL_TRAINING_STEPS = 20000
FIX_TRAINING_STEPS = 20000
TEST_SPLITS = [
  [True, False, False, False],
  [False, True, False, False],
  [False, False, True, False],
  [False, False, False, True],
  [False, False, True, True],
  [True, True, False, False],
  [False, True, True, False],
  [True, True, True, False],
  [True, True, True, True],
]
def run_experiment():
  full_model = build_model(trainable_last_layer=True)
  fixed_model = build_model(trainable_last_layer=False)
  fixed_updates = get_fixed_updates(full_model, fixed_model)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())

  # train full model
  run_training(sess, full_model, FULL_TRAINING_STEPS, True)

  # test full model
  print('Testing full_model after {} steps'.format(FULL_TRAINING_STEPS))
  for splits in TEST_SPLITS:
    run_test(sess, full_model, True, active_splits=splits)

  # copy softmax layer to fixed model
  sess.run(fixed_updates)

  # train full model more, this time with inactive sensors
  run_training(sess, full_model, FULL_TRAINING_STEPS, False)

  # test full model again
  print('Testing full_model trained with partial training data '
      'after {} steps'.format(FULL_TRAINING_STEPS))
  for splits in TEST_SPLITS:
    run_test(sess, full_model, False, active_splits=splits)

  # train fixed_model
  run_training(sess, fixed_model, FIX_TRAINING_STEPS, False)

  print('Testing fixed_model trained with partial training data '
      'after {} steps'.format(FIX_TRAINING_STEPS))
  for splits in TEST_SPLITS:
    run_test(sess, fixed_model, False, active_splits=splits)

if __name__ == '__main__':
  run_experiment()
