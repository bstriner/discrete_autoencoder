class Layer(object):
    def __init__(self, params, non_trainable_weights):
        self.params = params
        self.non_trainable_weights = non_trainable_weights

    def call(self, x):
        return x, []

    def call_validation(self, x):
        out, updates = self.call(x)
        return out
