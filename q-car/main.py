import tensorflow as tf

from td_model import TdModel

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = TdModel(sess)
        model.bootstrap()
        model.train()
