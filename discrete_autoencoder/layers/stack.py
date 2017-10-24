import itertools


class Stack(object):
    def __init__(self, layers):
        self.layers = layers
        self.params = list(itertools.chain.from_iterable(layer.params for layer in self.layers))
        self.non_trainable_weights = list(
            itertools.chain.from_iterable(layer.non_trainable_weights for layer in self.layers))
        
    def call(self, x):
        out = x
        updates = []
        for layer in self.layers:
            out, u = layer.call(out)
            updates += u
        return out, updates

    def call_validation(self, x):
        out = x
        for layer in self.layers:
            out = layer.call_validation(out)
        return out
