from trainable_models.model import NeuralNetwork
from trainable_models.vgg_like import VGGLikeNetwork

playground_config = dict()

playground_config['model_to_use'] = NeuralNetwork()
playground_config['data_folder'] = 'data'

playground_config['model_name'] = 'test_nn'
playground_config['save_location'] = 'saved_models/' + playground_config['model_name'] + ".pth"