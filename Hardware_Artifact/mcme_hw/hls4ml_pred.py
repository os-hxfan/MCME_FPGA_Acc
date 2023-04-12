#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
import keras 
from converter.keras.nn2bnn import MonteCarloDropout
import sys
import numpy as np 
import hls4ml
from converter.keras.train import mnist_data
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import argparse 
import os
import tensorflow_probability as tfp
import tensorflow as tf
from converter.keras.Dropouts import BayesianDropout

from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np


def convert_pred(args):
    co = {"BayesianDropout": BayesianDropout}
    _add_supported_quantized_objects(co)
    model = load_model(args.load_model + '.h5', custom_objects=co)
    model = MonteCarloDropout(model, nSamples=1, num=0, p=args.dropout_rate) # The Bayesian conversion according to the number of layers is deprecated after FPGA submission
    import hls4ml
    #import plotting

    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'])
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

    #First, the baseline model
    hls_config = hls4ml.utils.config_from_keras_model(model.model, granularity='name')

    # Set the precision and reuse factor for the full model

    # hls_config['Model']['Precision'] = 'ap_fixed<16,3>'
    hls_config['Model']['ReuseFactor'] = 90000000
    hls_config['Model']['Strategy'] = 'Resource'
    hls_config['Model']['BramFactor'] = 50000
    hls_config['Model']['MergeFactor'] = 1

    # Create an entry for each layer, here you can for instance change the strategy for a layer to 'resource'
    # or increase the reuse factor individually for large layers.
    # In this case, we designed the model to be small enough for a fully parallel implementation
    # so we use the latency strategy and reuse factor of 1 for all layers.
    for Layer in hls_config['LayerName'].keys():
        #hls_config['LayerName'][Layer]['Strategy'] = 'Latency'
        #hls_config['LayerName'][Layer]['ReuseFactor'] = 1
        hls_config['LayerName'][Layer]['Strategy'] = 'Resource'
        hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000
    #If you want best numerical performance for high-accuray models, while the default latency strategy is faster but numerically more unstable
    hls_config['LayerName']['softmax']['Strategy'] = 'Stable'
    #plotting.print_dict(hls_config)

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model.model
    cfg['OutputDir']  = args.load_model + '/'
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'
    cfg['Bayes'] = True

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    from train_qkeras_mcme import get_dataset
    data_dict = get_dataset(args)

    x_test_reduced = data_dict["x_test"][:50]
    y_test_reduced = data_dict["y_test"][:50]

    from sklearn.metrics import accuracy_score
    y_prob        = model.predict(x_test_reduced)
    # print (y_prob)
    # print ( tf.equal(y_prob[0], y_prob[1]))
    print (y_prob.dtype)
    ece_keras = tfp.stats.expected_calibration_error(num_bins=args.num_bins, 
        logits=y_prob, labels_true=np.argmax(y_test_reduced,axis=1), labels_predicted=np.argmax(y_prob,axis=1))
    accuracy_keras = float(accuracy_score (np.argmax(y_test_reduced,axis=1), np.argmax(y_prob,axis=1)))
    print("Accuracy Keras:  {}, ECE Keras {}".format(accuracy_keras, ece_keras))

    y_prob_hls4ml = hls_model.predict(np.ascontiguousarray(x_test_reduced))

    if isinstance(y_prob_hls4ml, list):
        y_prob_hls4ml = sum(y_prob_hls4ml) / len(y_prob_hls4ml)

    ece_hls4ml = tfp.stats.expected_calibration_error(num_bins=args.num_bins, 
        logits=y_prob_hls4ml, labels_true=np.argmax(y_test_reduced,axis=1), labels_predicted=np.argmax(y_prob_hls4ml,axis=1))
    accuracy_hls4ml = float(accuracy_score (np.argmax(y_test_reduced,axis=1), np.argmax(y_prob_hls4ml,axis=1)))

    print("Accuracy hls4ml: {}, ECE hls4ml {} ".format(accuracy_hls4ml, ece_hls4ml))


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist", type=str, required=True, help="Name of dataset")
    parser.add_argument("--gpus", default="0,1", type=str, required=True, help="GPUs id, separated by comma without space, e.g, 0,1,2")
    parser.add_argument("--load_model", default=None, type=str, help="Name of load model")
    parser.add_argument("--num_bins", default=10, type=int, help="The number of bins while calculating ECE")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="The dropout rate")

    args = parser.parse_args()

    # Set GPU environment
    gpus = args.gpus.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

    convert_pred(args)
