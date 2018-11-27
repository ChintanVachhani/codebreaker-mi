import tensorflow as tf

def normalize(inputs, 
              type="bn",
              decay=.99,
              is_training=True, 
              activation_fn=None,
              scope="normalize"):
    if type == "bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=None,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   zero_debias_moving_mean=True,
                                                   fused=True)
            # restore original shape
            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)
        else:  # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   updates_collections=None,
                                                   is_training=is_training,
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
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [1], keep_dims=True)
            gamma = tf.get_variable("gamma",
                                    shape=params_shape,
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer)
            beta = tf.get_variable("beta",
                                   shape=params_shape,
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            normalized = (inputs - mean) / tf.sqrt(variance + 1e-8)
            outputs = normalized * gamma + beta

    else:  # None
        outputs = inputs

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def conv(inputs,
         filters=None,
         size=1,
         rate=1,
         padding="SAME",
         use_bias=False,
         is_training=True,
         activation_fn=None,
         decay=0.99,
         norm_type=None,
         scope="conv",
         reuse=None):
    ndims = inputs.get_shape().ndims
    conv_fn = tf.layers.conv1d if ndims == 3 else tf.layers.conv2d

    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            assert ndims == 3, "if causal is true, the rank must be 3."
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding,
                  "use_bias": use_bias, "reuse": reuse}
        outputs = conv_fn(**params)
        outputs = normalize(outputs, type=norm_type, decay=decay,
                            is_training=is_training, activation_fn=activation_fn)
    return outputs