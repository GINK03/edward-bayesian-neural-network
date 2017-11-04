import json
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


def neural_network(X):
  h = tf.tanh(tf.matmul(X, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])

ed.set_seed(42)

N = 100  # number of data points
D = 2   # number of features

# DATA
y_train, X_train = json.loads( open('./data.json').read() )
y_train, X_train = np.array(y_train), np.array(X_train)
#X_train, y_train = build_toy_dataset(N)
print( X_train.shape  )
print( y_train.shape  )
# MODEL
with tf.name_scope("model"):
  W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]), name="W_0")
  W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
  W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_2")
  b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
  b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
  b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

  X = tf.placeholder(tf.float32, [N, D], name="X")
  y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(N), name="y")

# INFERENCE
with tf.name_scope("posterior"):
  qWs = []
  for index, inputs, outputs in [(0, D, 10), (1, 10, 10), (2, 10, 1)]:
    with tf.name_scope("qW_%d"%index):
      qW = Normal(loc=tf.Variable(tf.random_normal([inputs, outputs]), name="loc"),
                    scale=tf.nn.softplus(
                        tf.Variable(tf.random_normal([inputs, outputs]), name="scale")))
      qWs.append(qW)
  qbs = []
  for index, outputs in [(0, 10), (1, 10), (2, 1)]:
    with tf.name_scope("qb_%d"%index):
      qb = Normal(loc=tf.Variable(tf.random_normal([outputs]), name="loc"),
                    scale=tf.nn.softplus(
                        tf.Variable(tf.random_normal([outputs]), name="scale")))
      qbs.append(qb)

inference = ed.KLqp({W_0: qWs[0], b_0: qbs[0],
                     W_1: qWs[1], b_1: qbs[1],
                     W_2: qWs[2], b_2: qbs[2] }, data={X: X_train, y: y_train})
inference.run(logdir='log')

def sample():
  sw0 = W_0.sample(1).eval().tolist()[0]
  sw1 = W_1.sample(1).eval().tolist()[0]
  sw2 = W_2.sample(1).eval().tolist()[0]
  sb0 = b_0.sample(1).eval().tolist()[0] 
  sb1 = b_1.sample(1).eval().tolist()[0] 
  sb2 = b_2.sample(1).eval().tolist()[0] 
  print(sw0) 
  X = tf.placeholder(tf.float32, [N, D], name="X")
  X = np.array(X_train, dtype=np.float32 )
  h = tf.tanh(tf.matmul(X, sw0) + sb0)
  h = tf.tanh(tf.matmul(h, sw1) + sb1)
  h = tf.matmul(h, sw2) + sb2
  h = tf.reshape(h, [-1])
  y = Normal(loc=h, scale=0.1 * tf.ones(N), name="y")
  sample = y.sample(1).eval()
  print('MMMorimori', sample)



sample()

