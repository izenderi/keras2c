"""keras2c_main.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
import keras
from keras2c.make_test_suite import make_test_suite
from keras2c.check_model import check_model
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2c.weights2c import Weights2C
from keras2c.layer2c import Layers2C


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, file, function_name):
    model_inputs, model_outputs = get_model_io_names(model)

    s = '#include <stdio.h> \n#include <stddef.h> \n#include <math.h> \n#include <string.h> \n'
    s += '#include <stdarg.h> \n#include "k2c_include.h" \n'
    s += '\n \n'
    file.write(s)

    print('Gathering Weights')
    stack_vars, malloc_vars = Weights2C(model).write_weights()
    layers = Layers2C(model).write_layers()

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ') { \n\n'

    function_init_signature = 'void ' + function_name + '_initialize('
    function_init_signature += ','.join(['float* ' +
                                         key for key in malloc_vars.keys()])
    function_init_signature += ') { \n\n'

    function_term_signature = 'void ' + function_name + '_terminate('
    function_term_signature += ','.join(['float* ' +
                                         key for key in malloc_vars.keys()])
    function_term_signature += ') { \n\n'

    file.write(function_signature)
    file.write(stack_vars)
    file.write(layers)
    file.write('\n }')


# keras2c
def k2c(model, function_name, num_tests=10):

    function_name = str(function_name)
    filename = function_name + '.h'
    if isinstance(model, str):
        model = keras.models.load_model(str(model_filepath))
    elif not isinstance(model, (keras.models.Model,
                                keras.engine.training.Model)):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    print('All checks passed')

    file = open(filename, "x+")
    model2c(model, file, function_name)
    file.close()
    make_test_suite(model, function_name, num_tests)
    print("Done \n C code is in '" + function_name +
          ".h' and tests are in '" + function_name + "_test_suite.c'")
