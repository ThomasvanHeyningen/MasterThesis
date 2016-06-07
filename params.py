class Params:

    epochs = None
    learning_rate = None
    mini_batch = None
    tumor_size = None
    patch_size = None
    batch_size = None

    def __init__(self):
        self.epochs = 1000
        self.learning_rate = 0.0001
        self.mini_batch = 15
        self.tumor_size = 25
        self.patch_size = 32
        self.batch_size = 15
