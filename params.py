class Params:

    epochs = None
    learning_rate = None
    mini_batch = None
    tumor_size = None
    patch_size = None
    batch_size = None
    network = None  # standard/VGG
    validation_size = None
    test_size = None
    mode = None  # 0 for train, 1 for test

    def __init__(self):
        self.epochs = 100
        self.learning_rate = 0.00001
        self.tumor_size = 25
        self.patch_size = 32
        self.batch_size = 1
        self.network = 'VGG'
        self.validation_size = 80
        self.test_size = 80
        self.mode = 0
