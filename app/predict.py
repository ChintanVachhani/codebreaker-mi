import tensorflow as tf
import numpy as np
from .train import Graph
from .hyperparams import Hyperparams as hp

def predict(quiz):
    x = np.zeros((1, 9 * 9), np.float32)
    for j, (q) in enumerate(quiz):
        x[0, j] = q
    x = np.reshape(x, (-1, 9, 9))
    g = Graph(is_training=False)
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            sv.saver.restore(session, tf.train.latest_checkpoint(hp.logdir))
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
            for xx in (y.reshape(-1, 9*9)):
                return "{}".format("".join(str(num) for num in xx))