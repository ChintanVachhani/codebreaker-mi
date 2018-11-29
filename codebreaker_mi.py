import tensorflow as tf
import numpy as np


class Param:
    # data
    trainFilePath = 'sudoku.csv'
    testFilePath = 'test.csv'

    # model
    numBlocks = 10
    numFilters = 512
    filterSize = 3

    # training scheme
    lr = 0.0001
    modelDir = "model"
    batchSize = 20
    numEpochs = 3


class DataLoad:
    @staticmethod
    def loadData(type="train"):
        filePath = Param.trainFilePath if type == "train" else Param.testFilePath
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

    def getData(self):
        X, Y = self.loadData(type="train")

        # Create Queues
        inputQueues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.float32),
                                                     tf.convert_to_tensor(Y, tf.int32)])

        # create batch queues
        x, y = tf.train.shuffle_batch(inputQueues,
                                      num_threads=8,
                                      batch_size=Param.batchSize,
                                      capacity=Param.batchSize * 64,
                                      min_after_dequeue=Param.batchSize * 32,
                                      allow_smaller_final_batch=False)
        # calc total batch count
        numBatch = len(X) // Param.batchSize

        return x, y, numBatch


class Modules:
    @staticmethod
    def normalize(inputs,
                  type="bn",
                  decay=.99,
                  isTraining=True,
                  activationFn=None,
                  scope="normalize"):
        if type == "bn":
            inputsShape = inputs.get_shape()
            inputsRank = inputsShape.ndims

            # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
            # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
            if inputsRank in [2, 3, 4]:
                if inputsRank == 2:
                    inputs = tf.expand_dims(inputs, axis=1)
                    inputs = tf.expand_dims(inputs, axis=2)
                elif inputsRank == 3:
                    inputs = tf.expand_dims(inputs, axis=1)

                outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                       decay=decay,
                                                       center=True,
                                                       scale=True,
                                                       activation_fn=None,
                                                       updates_collections=None,
                                                       is_training=isTraining,
                                                       scope=scope,
                                                       zero_debias_moving_mean=True,
                                                       fused=True)
                # restore original shape
                if inputsRank == 2:
                    outputs = tf.squeeze(outputs, axis=[1, 2])
                elif inputsRank == 3:
                    outputs = tf.squeeze(outputs, axis=1)
            else:  # fallback to naive batch norm
                outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                       decay=decay,
                                                       center=True,
                                                       scale=True,
                                                       activation_fn=activationFn,
                                                       updates_collections=None,
                                                       is_training=isTraining,
                                                       scope=scope,
                                                       fused=False)
        elif type == "ln":
            outputs = tf.contrib.layers.layer_norm(inputs=inputs,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=None,
                                                   scope=scope)
        elif type == "in":  # instance normalization
            with tf.variable_scope(scope):
                inputsShape = inputs.get_shape()
                ParamsShape = inputsShape[-1:]

                mean, variance = tf.nn.moments(inputs, [1], keep_dims=True)
                gamma = tf.get_variable("gamma",
                                        shape=ParamsShape,
                                        dtype=tf.float32,
                                        initializer=tf.ones_initializer)
                beta = tf.get_variable("beta",
                                       shape=ParamsShape,
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer)
                normalized = (inputs - mean) / tf.sqrt(variance + 1e-8)
                outputs = normalized * gamma + beta

        else:  # None
            outputs = inputs

        if activationFn is not None:
            outputs = activationFn(outputs)

        return outputs

    def conv(self, inputs,
             filters=None,
             size=1,
             rate=1,
             padding="SAME",
             useBias=False,
             isTraining=True,
             activationFn=None,
             decay=0.99,
             normType=None,
             scope="conv",
             reuse=None):
        ndims = inputs.get_shape().ndims
        convFn = tf.layers.conv1d if ndims == 3 else tf.layers.conv2d

        with tf.variable_scope(scope):
            if padding.lower() == "causal":
                assert ndims == 3, "if causal is true, the rank must be 3."
                # pre-padding for causality
                padLen = (size - 1) * rate  # padding size
                inputs = tf.pad(inputs, [[0, 0], [padLen, 0], [0, 0]])
                padding = "valid"

            if filters is None:
                filters = inputs.get_shape().as_list[-1]

            Params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                      "dilation_rate": rate, "padding": padding,
                      "use_bias": useBias, "reuse": reuse}
            outputs = convFn(**Params)
            outputs = self.normalize(outputs, type=normType, decay=decay,
                                     isTraining=isTraining, activationFn=activationFn)
        return outputs


class Graph(object):
    def __init__(self, isTraining=True):
        self.graph = tf.Graph()
        self.m = Modules()
        with self.graph.as_default():
            # inputs
            if isTraining:
                self.x, self.y, self.num_batch = DataLoad.getBatchData()  # (N, 9, 9)
            else:
                self.x = tf.placeholder(tf.float32, (None, 9, 9))
                self.y = tf.placeholder(tf.int32, (None, 9, 9))
            self.enc = tf.expand_dims(self.x, axis=-1)  # (N, 9, 9, 1)
            self.istarget = tf.to_float(tf.equal(self.x, tf.zeros_like(self.x)))  # 0: blanks

            # network
            for i in range(Param.numBlocks):
                with tf.variable_scope("conv2d_{}".format(i)):
                    self.enc = self.m.conv(inputs=self.enc,
                                      filters=Param.numFilters,
                                      size=Param.filterSize,
                                      isTraining=isTraining,
                                      normType="bn",
                                      activationFn=tf.nn.relu)

            # outputs
            self.logits = self.m.conv(self.enc, 10, 1, scope="logits")  # (N, 9, 9, 1)
            self.probs = tf.reduce_max(tf.nn.softmax(self.logits), axis=-1)  # ( N, 9, 9)
            self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))  # ( N, 9, 9)

            # accuracy
            self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
            self.acc = tf.reduce_sum(self.hits) / (tf.reduce_sum(self.istarget) + 1e-8)
            tf.summary.scalar("acc", self.acc)

            if isTraining:
                # Loss
                self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
                self.loss = tf.reduce_sum(self.ce * self.istarget) / (tf.reduce_sum(self.istarget))

                # Training Scheme
                self.globalStep = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=Param.lr)
                self.trainOp = self.optimizer.minimize(self.loss, global_step=self.globalStep)
                tf.summary.scalar("loss", self.loss)

            self.merged = tf.summary.merge_all()


class PuzzleSolver:
    def predict(puzzle):
        x = np.reshape(np.asarray(puzzle, np.float32), (1, 9, 9))
        g = Graph(isTraining=False)
        with g.graph.as_default():
            sv = tf.train.Supervisor()
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
                sv.saver.restore(session, tf.train.latest_checkpoint(Param.modelDir))
                import copy
                sol = copy.copy(x)
                while 1:
                    istarget, probs, preds = session.run([g.istarget, g.probs, g.preds], {g.x: sol})
                    probs = probs.astype(np.float32)
                    preds = preds.astype(np.float32)

                    probs *= istarget
                    preds *= istarget

                    probs = np.reshape(probs, (-1, 9 * 9))
                    preds = np.reshape(preds, (-1, 9 * 9))

                    sol = np.reshape(sol, (-1, 9 * 9))
                    maxprob_ids = np.argmax(probs, axis=1)
                    maxprobs = np.max(probs, axis=1, keepdims=False)
                    for j, (maxprob_id, maxprob) in enumerate(zip(maxprob_ids, maxprobs)):
                        if maxprob != 0:
                            sol[j, maxprob_id] = preds[j, maxprob_id]
                    sol = np.reshape(sol, (-1, 9, 9))
                    sol = np.where(x == 0, sol, x)

                    if np.count_nonzero(sol) == sol.size: break
                y = sol.astype(np.int32)
                return y[0].tolist()
