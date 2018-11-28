import tensorflow as tf
import numpy as np
from .params import params as param

def load_data(type="train"):
    filePath = param.trainFilePath if type == "train" else param.testFilePath
    lines = open(filePath, 'r').read().splitlines()[1:]
    nsamples = len(lines)

    X = np.zeros((nsamples, 9 * 9), np.float32)
    Y = np.zeros((nsamples, 9 * 9), np.int32)

    for i, line in enumerate(lines):
        quiz, solution = line.split(",")
        print(quiz)
        for j, (q, s) in enumerate(zip(quiz, solution)):
            X[i, j], Y[i, j] = q, s

    X = np.reshape(X, (-1, 9, 9))
    Y = np.reshape(Y, (-1, 9, 9))
    return X, Y
        
def get_batch_data():
    X, Y = load_data(type="train")

    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.float32),
                                                  tf.convert_to_tensor(Y, tf.int32)])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=param.batch_size,
                                  capacity=param.batch_size * 64,
                                  min_after_dequeue=param.batch_size * 32,
                                  allow_smaller_final_batch=False)
    # calc total batch count
    num_batch = len(X) // param.batch_size

    return x, y, num_batch
