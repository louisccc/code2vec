class Doc2VecConfig:
    def __init__(self):
        pass

        
class Code2VecConfig:

    def __init__(self):
        # hyperparameters used in tf dataset training. 

        self.epoch = 500
        self.training_batch_size = 256
        self.testing_batch_size = 128

        self.dropout_factor = 0.5
        self.learning_rate = 0.005
        self.embedding_size = 50
        self.code_embedding_size = 50

        self.max_contexts = 200