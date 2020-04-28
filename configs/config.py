from argparse import ArgumentParser
from pathlib import Path
import pprint


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


class GraphASTArgParser:
    '''
        Argument Parser for graphast script.
    '''
    
    def __init__(self):
        self.parser = ArgumentParser(description='The parameters for graphast method.')
        
        self.parser.add_argument('-ip', dest='input_path',
                                 default='../test/fashion_mnist', type=str,
                                 help='Path to the source code. Default: ../test/fashion_mnist')
        self.parser.add_argument('-r', dest='recursive', 
                                 action='store_true',
                                 help='Recursively apply graphast method on all papers in the input path.')
        self.parser.set_defaults(recursive=False)
        self.parser.add_argument('-dp', '--dest_path',
                                 default='../graphast_output', type=str,
                                 help='Path to save output files.')
        self.parser.add_argument('-res', dest='resolution',
                                 default='function', type=str,
                                 help='Processing resolution of the method: function or method. Default: function.')

    def get_args(self, args):
        return self.parser.parse_args(args)


class GraphASTConfig:
    '''
        Config class for graphast method.
    '''

    def __init__(self, arg):
        self.input_path = Path(arg.input_path).resolve()
        self.recursive = arg.recursive
        self.dest_path = Path(arg.dest_path).resolve()
        self.resolution = arg.resolution

    def dump(self):
        pprint.pprint(self.__dict__)


class PyKG2VecArgParser (KGEArgParser):
    """The class implements the argument parser for the pykg2vec script."""

    def __init__(self):
        super().__init__()
        self.general_group.set_defaults(dataset_name='lightweight')
        self.general_group.add_argument('-tp', dest='triples_path', default=None, type=str, 
                                        help='The path to output triples from lightweight method.')