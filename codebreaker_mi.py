import tensorflow as tf
import numpy as np

"""
    Hyperparams for the tensorflow training and prediction 
"""
class Param:
    numBlocks = 10
    numFilters = 512
    filterSize = 3
    lr = 0.0001
    modelDir = "model"
    batchSize = 20
    numEpochs = 3
"""
    Tensorflow modules norm and conv
"""
class Modules:
    @staticmethod
    def norm(inputs,
                  type="bn",
                  decay=.99,
                  activationFn=None,
                  scope="normalize"):
        if type == "bn":
            inputsShape = inputs.get_shape()
            inputsRank = inputsShape.ndims

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
                                                       is_training=False,
                                                       scope=scope,
                                                       zero_debias_moving_mean=True,
                                                       fused=True)
                if inputsRank == 2:
                    outputs = tf.squeeze(outputs, axis=[1, 2])
                elif inputsRank == 3:
                    outputs = tf.squeeze(outputs, axis=1)
            else:
                outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                       decay=decay,
                                                       center=True,
                                                       scale=True,
                                                       activation_fn=activationFn,
                                                       updates_collections=None,
                                                       is_training=False,
                                                       scope=scope,
                                                       fused=False)
        elif type == "ln":
            outputs = tf.contrib.layers.layer_norm(inputs=inputs,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=None,
                                                   scope=scope)
        elif type == "in":
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
                afterNorm = (inputs - mean) / tf.sqrt(variance + 1e-8)
                outputs = afterNorm * gamma + beta

        else:
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
             activationFn=None,
             decay=0.99,
             normType=None,
             scope="conv",
             reuse=None):
        ndims = inputs.get_shape().ndims
        convFn = tf.layers.conv1d if ndims == 3 else tf.layers.conv2d

        with tf.variable_scope(scope):
            if padding.lower() == "causal":
                assert ndims == 3, ""
                padLen = (size - 1) * rate
                inputs = tf.pad(inputs, [[0, 0], [padLen, 0], [0, 0]])
                padding = "valid"

            if filters is None:
                filters = inputs.get_shape().as_list[-1]

            Params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                      "dilation_rate": rate, "padding": padding,
                      "use_bias": useBias, "reuse": reuse}
            outputs = convFn(**Params)
            outputs = self.norm(outputs, type=normType, decay=decay, activationFn=activationFn)
        return outputs

class TFGraph(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.m = Modules()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (None, 9, 9))
            self.y = tf.placeholder(tf.int32, (None, 9, 9))
            self.enc = tf.expand_dims(self.x, axis=-1)
            self.checkTarget = tf.to_float(tf.equal(self.x, tf.zeros_like(self.x)))
            for i in range(Param.numBlocks):
                with tf.variable_scope("conv2d_{}".format(i)):
                    self.enc = self.m.conv(inputs=self.enc,
                                      filters=Param.numFilters,
                                      size=Param.filterSize,
                                      normType="bn",
                                      activationFn=tf.nn.relu)
            self.logits = self.m.conv(self.enc, 10, 1, scope="logits")
            self.probability = tf.reduce_max(tf.nn.softmax(self.logits), axis=-1)
            self.prediction = tf.to_int32(tf.argmax(self.logits, dimension=-1))
            self.hits = tf.to_float(tf.equal(self.prediction, self.y)) * self.checkTarget
            self.acc = tf.reduce_sum(self.hits) / (tf.reduce_sum(self.checkTarget) + 1e-8)
            tf.summary.scalar("acc", self.acc)
            self.merged = tf.summary.merge_all()

class PuzzleSolver:
    def predict(puzzle):
        x = np.reshape(np.asarray(puzzle, np.float32), (1, 9, 9))
        g = TFGraph()
        with g.graph.as_default():
            sv = tf.train.Supervisor()
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
                sv.saver.restore(session, tf.train.latest_checkpoint(Param.modelDir))
                import copy
                sol = copy.copy(x)
                while 1:
                    checkTarget, probability, prediction = session.run([g.checkTarget, g.probability, g.prediction], {g.x: sol})
                    probability = probability.astype(np.float32)
                    prediction = prediction.astype(np.float32)

                    probability *= checkTarget
                    prediction *= checkTarget

                    probability = np.reshape(probability, (-1, 9 * 9))
                    prediction = np.reshape(prediction, (-1, 9 * 9))

                    sol = np.reshape(sol, (-1, 9 * 9))
                    maxprobability_id = np.argmax(probability, axis=1)
                    maxprobability = np.max(probability, axis=1, keepdims=False)
                    for j, (maxprobability_id, maxprobability) in enumerate(zip(maxprobability_id, maxprobability)):
                        if maxprobability != 0:
                            sol[j, maxprobability_id] = prediction[j, maxprobability_id]
                    sol = np.reshape(sol, (-1, 9, 9))
                    sol = np.where(x == 0, sol, x)

                    if np.count_nonzero(sol) == sol.size: break
                y = sol.astype(np.int32)
                return y[0].tolist()
