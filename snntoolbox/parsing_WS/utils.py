# -*- coding: utf-8 -*-
"""Functions common to input model parsers.

The core of this module is an abstract base class extracts an input model
written in some neural network library and prepares it for further processing
in the SNN toolbox.

.. autosummary::
    :nosignatures:

    AbstractModelParser

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format.

Other functions help navigate through the network in order to explore network
connectivity and layer attributes:

.. autosummary::
    :nosignatures:

    get_type
    has_weights
    get_fanin
    get_fanout
    get_inbound_layers
    get_inbound_layers_with_params
    get_inbound_layers_without_params
    get_outbound_layers
    get_outbound_activation

@author: rbodo
"""
import json

import pickle

from abc import abstractmethod

from tensorflow import keras
import whetstone
from whetstone.utils.layer_utils import load_model as load_WS_model
import numpy as np

IS_CHANNELS_FIRST = keras.backend.image_data_format() == 'channels_first'


class AbstractModelParser:
    """Abstract base class for neural network model parsers.

    Parameters
    ----------

    input_model
        The input network object.
    config: configparser.Configparser
        Contains the toolbox configuration for a particular experiment.

    Attributes
    ----------

    input_model: dict
        The input network object.
    config: configparser.Configparser
        Contains the toolbox configuration for a particular experiment.
    _layer_list: list[dict]
        A list where each entry is a dictionary containing layer
        specifications. Obtained by calling `parse`. Used to build new, parsed
        Keras model.
    _layer_dict: dict
        Maps the layer names of the specific input model library to our
        standard names (currently Keras).
    parsed_model: keras.models.Model
        The parsed model.
    """

    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self._layer_list = []
        self._layer_dict = {}
        self.parsed_model = None

    def parse_subnet(self, layers, idx, prev_out_idx=None, in_layers=None, out_layers=None, repair=None, special_relu=[], \
        remove_from_relu = ('max_value', 'negative_slope', 'threshold')):
        name_map = {}
        out_links = [None]*len(out_layers)
        is_output = False
        need_rewire = False

        if prev_out_idx is not None and in_layers is not None:
            if len(prev_out_idx) != len(in_layers):
                raise ValueError("prev_out_idx and in_layers must be the same size.")
        
        for k_,layer in enumerate(layers):
            layer_type = self.get_type(layer)

            is_output = layer in out_layers

            if repair is not None:
                need_rewire = layer in repair

            if prev_out_idx is None or layer not in in_layers:
                if need_rewire:
                    inbound = [self._layer_list[-1]['name']]
                else:
                    inbound = self.get_inbound_names(layer, name_map)
            else:
                inbound = []
                for j,in_layer in enumerate(in_layers):
                    if layer==in_layer:
                        inbound.append(self._layer_list[prev_out_idx[j]]['name'])

            attributes = self.initialize_attributes(layer)


            if layer_type=='Activation':
                attributes.pop('activation')
                activation_str = self.get_activation(layer)              
                if activation_str == 'relu':
                    layer_type = 'Spiking_BRelu'
                elif activation_str == 'sigmoid':
                    layer_type = 'Spiking_Sigmoid'
                elif activation_str == 'linear':
                    continue

            if layer_type=='ReLU':
                layer_type = 'Spiking_BRelu'
                for entry in remove_from_relu:
                    attributes.pop(entry, None)                 

            attributes.update({'layer_type': layer_type,
                               'name': self.get_name(layer, idx),
                               'inbound': inbound})

            if layer_type == 'UpSampling2D':
                attributes.update({'size': layer.size})
            
            if layer_type in {'Conv1D', 'Conv2D', 'Dense', 'Sparse', 'SparseConv2D'}:
                attributes['parameters'] = list(layer.get_weights())

            self._layer_list.append(attributes)

            # Map layer index to layer id. Needed for inception modules.
            name_map[str(id(layer))] = idx

            if is_output:
                # get output layers references
                for i,out_layer in enumerate(out_layers):
                    if layer==out_layer:
                        out_links[i]=idx

            print("Parsing layer {}.".format(layer_type))
            idx += 1
        print('')

        return idx,out_links


    @abstractmethod
    def get_layer_iterable(self):
        """Get an iterable over the layers of the network.

        Returns
        -------

        layers: list
        """

        pass

    @abstractmethod
    def get_type(self, layer):
        """Get layer class name.

        Returns
        -------

        layer_type: str
            Layer class name.
        """

        pass

    @abstractmethod
    def get_batchnorm_parameters(self, layer):
        """Get the parameters of a batch-normalization layer.

        Returns
        -------

        mean, var_eps_sqrt_inv, gamma, beta, axis: tuple

        """

        pass

    def get_inbound_layers_with_parameters(self, layer):
        """Iterate until inbound layers are found that have parameters.

        Parameters
        ----------

        layer:
            Layer

        Returns
        -------

        : list
            List of inbound layers.
        """

        inbound = layer
        while True:
            inbound = self.get_inbound_layers(inbound)
            if len(inbound) == 1:
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
            else:
                result = []
                for inb in inbound:
                    if self.has_weights(inb):
                        result.append(inb)
                    else:
                        result += self.get_inbound_layers_with_parameters(inb)
                return result

    def get_inbound_names(self, layer, name_map):
        """Get names of inbound layers.

        Parameters
        ----------

        layer:
            Layer
        name_map: dict
            Maps the name of a layer to the `id` of the layer object.

        Returns
        -------

        : list
            The names of inbound layers.

        """

        inbound = self.get_inbound_layers(layer)
        for ib in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[ib]) in self.layers_to_skip:
                    inbound[ib] = self.get_inbound_layers(inbound[ib])[0]
                else:
                    break
        if len(self._layer_list) == 0 or \
                any([self.get_type(inb) == 'InputLayer' for inb in inbound]):
            return [self.input_layer_name]
        else:
            inb_idxs = [name_map[str(id(inb))] for inb in inbound]
            return [self._layer_list[i]['name'] for i in inb_idxs]

    @abstractmethod
    def get_inbound_layers(self, layer):
        """Get inbound layers of ``layer``.

        Returns
        -------

        inbound: Sequence
        """

        pass

    @property
    def layers_to_skip(self):
        """
        Return a list of layer names that should be skipped during conversion
        to a spiking network.

        Returns
        -------

        self._layers_to_skip: List[str]
        """

        # Todo: We should get this list from some central place like the
        #       ``config_defaults`` file.
        return ['BatchNormalization',
                'Activation',
                'Dropout',
                'ReLU',
                'ActivityRegularization',
                'GaussianNoise']

    @abstractmethod
    def has_weights(self, layer):
        """Return ``True`` if ``layer`` has weights."""

        pass

    def initialize_attributes(self, layer=None):
        """
        Return a dictionary that will be used to collect all attributes of a
        layer. This dictionary can then be used to instantiate a new parsed
        layer.
        """

        return {}

    @abstractmethod
    def get_input_shape(self):
        """Get the input shape of a network, not including batch size.

        Returns
        -------

        input_shape: tuple
            Input shape.
        """

        pass

    def get_batch_input_shape(self):
        """Get the input shape of a network, including batch size.

        Returns
        -------

        batch_input_shape: tuple
            Batch input shape.
        """

        input_shape = tuple(self.get_input_shape())
        batch_size = self.config.getint('simulation', 'batch_size')
        return (batch_size,) + input_shape

    def get_name(self, layer, idx, layer_type=None):
        """Create a name for a ``layer``.

        The format is <layer_num><layer_type>_<layer_shape>.

        >>> # Name of first convolution layer with 32 feature maps and
        >>> # dimension 64x64:
        "00Conv2D_32x64x64"
        >>> # Name of final dense layer with 100 units:
        "06Dense_100"

        Parameters
        ----------

        layer:
            Layer.
        idx: int
            Layer index.
        layer_type: Optional[str]
            Type of layer.

        Returns
        -------

        name: str
            Layer name.
        """

        if layer_type is None:
            layer_type = self.get_type(layer)

        try:
            output_shape = self.get_output_shape(layer)
            shape_string = ["{}x".format(x) for x in output_shape[1:]]
            shape_string[0] = "_" + shape_string[0]
            shape_string[-1] = shape_string[-1][:-1]
            shape_string = "".join(shape_string)
        except:
            shape_string = "MULT"

        num_str = self.format_layer_idx(idx)

        return num_str + layer_type + shape_string

    def format_layer_idx(self, idx):
        """Pad the layer index with the appropriate amount of zeros.

        The number of zeros used for padding is determined by the maximum index
        (i.e. the number of layers in the network).

        Parameters
        ----------

        idx: int
            Layer index.

        Returns
        -------

        num_str: str
            Zero-padded layer index.
        """

        max_idx = len(self.input_model.layers)
        return str(idx).zfill(len(str(max_idx)))

    @abstractmethod
    def get_output_shape(self, layer):
        """Get output shape of a ``layer``.

        Parameters
        ----------

        layer
            Layer.

        Returns
        -------

        output_shape: Sized
            Output shape of ``layer``.
        """

        pass

    def try_insert_flatten(self, layer, idx, name_map):
        output_shape = self.get_output_shape(layer)
        previous_layers = self.get_inbound_layers(layer)
        prev_layer_output_shape = self.get_output_shape(previous_layers[0])
        if len(output_shape) < len(prev_layer_output_shape) and \
                self.get_type(layer) not in {'Flatten', 'Reshape'} and \
                self.get_type(previous_layers[0]) != 'InputLayer':
            assert len(previous_layers) == 1, \
                "Layer to flatten must be unique."
            print("Inserting layer Flatten.")
            num_str = self.format_layer_idx(idx)
            shape_string = str(np.prod(prev_layer_output_shape[1:]))
            self._layer_list.append({
                'name': num_str + 'Flatten_' + shape_string,
                'layer_type': 'Flatten',
                'inbound': self.get_inbound_names(layer, name_map)})
            name_map['Flatten' + str(idx)] = idx
            return True
        else:
            return False

    @abstractmethod
    def parse_dense(self, layer, attributes):
        """Parse a fully-connected layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    @abstractmethod
    def parse_convolution(self, layer, attributes):
        """Parse a convolutional layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    @abstractmethod
    def parse_depthwiseconvolution(self, layer, attributes):
        """Parse a depthwise convolution layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    def parse_sparse(self, layer, attributes):
        pass

    def parse_sparse_convolution(self, layer, attributes):
        pass

    def parse_sparse_depthwiseconvolution(self, layer, attributes):
        pass

    @abstractmethod
    def parse_pooling(self, layer, attributes):
        """Parse a pooling layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    @abstractmethod
    def get_activation(self, layer):
        """Get the activation string of an activation ``layer``.

        Parameters
        ----------

        layer
            Layer

        Returns
        -------

        activation: str
            String indicating the activation of the ``layer``.

        """

        pass

    @abstractmethod
    def get_outbound_layers(self, layer):
        """Get outbound layers of ``layer``.

        Parameters
        ----------

        layer:
            Layer.

        Returns
        -------

        outbound: list
            Outbound layers of ``layer``.

        """

        pass

    @abstractmethod
    def parse_concatenate(self, layer, attributes):
        """Parse a concatenation layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass


    def build_parsed_RNet(self, out_refs, loss_fn, optimizer, input_shape, metrics=None):

        img_input = keras.layers.Input(
            shape = input_shape,
            name=self.input_layer_name)
        parsed_layers = {self.input_layer_name: img_input}
        print("Building parsed model...\n")
        for layer in self._layer_list:
            # Replace 'parameters' key with Keras key 'weights'
            if 'parameters' in layer:
                layer['weights'] = layer.pop('parameters')

            # Add layer
            layer_type = layer.pop('layer_type')
            layer.pop('dtype')
            if hasattr(keras.layers, layer_type):
                parsed_layer = getattr(keras.layers, layer_type)
            elif hasattr(whetstone.layers, layer_type):
                parsed_layer = getattr(whetstone.layers, layer_type)
            else:
                import keras_rewiring
                parsed_layer = getattr(keras_rewiring.sparse_layer, layer_type)

            inbound = [parsed_layers[inb] for inb in layer.pop('inbound')]
            if len(inbound) == 1:
                inbound = inbound[0]
            check_for_custom_activations(layer)
            parsed_layers[layer['name']] = parsed_layer(**layer)(inbound)

        print("Compiling parsed model...\n")
        self.parsed_model = keras.models.Model(
            inputs = img_input,
            outputs = [parsed_layers[self._layer_list[i]['name']] for i in out_refs]
            )
        self.parsed_model.save('temp.h5')
        self.parsed_model = load_WS_model('temp.h5')

        if metrics is not None:
            self.parsed_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
        else:
            self.parsed_model.compile(loss=loss_fn, optimizer=optimizer)

        self.parsed_model.summary()
        return self.parsed_model



    def evaluate(self, batch_size, num_to_test, x_test=None, y_test=None,
                 dataflow=None):
        """Evaluate parsed Keras model.

        Can use either numpy arrays ``x_test, y_test`` containing the test
        samples, or generate them with a dataflow
        (``keras.ImageDataGenerator.flow_from_directory`` object).

        Parameters
        ----------

        batch_size: int
            Batch size

        num_to_test: int
            Number of samples to test

        x_test: Optional[np.ndarray]

        y_test: Optional[np.ndarray]

        dataflow: keras.ImageDataGenerator.flow_from_directory
        """

        assert (x_test is not None and y_test is not None or dataflow is not
                None), "No testsamples provided."

        if x_test is not None:
            score = self.parsed_model.evaluate(x_test, y_test, batch_size,
                                               verbose=0)
        else:
            steps = int(num_to_test / batch_size)
            score = self.parsed_model.evaluate(dataflow, steps=steps)
        print("Top-1 accuracy: {:.2%}".format(score[1]))
        print("Top-5 accuracy: {:.2%}\n".format(score[2]))

        return score

    @property
    def input_layer_name(self):
        return 'input'


def absorb_bn_parameters(weight, bias, mean, var_eps_sqrt_inv, gamma, beta,
                         axis, image_data_format, is_depthwise=False):
    """
    Absorb the parameters of a batch-normalization layer into the previous
    layer.
    """

    axis = weight.ndim + axis if axis < 0 else axis

    print("Using BatchNorm axis {}.".format(axis))

    # Map batch norm axis from layer dimension space to kernel dimension space.
    # Assumes that kernels are shaped like
    # [height, width, num_input_channels, num_output_channels],
    # and layers like [batch_size, channels, height, width] or
    # [batch_size, height, width, channels].
    if weight.ndim == 4:

        channel_axis = 2 if is_depthwise else 3

        if image_data_format == 'channels_first':
            layer2kernel_axes_map = [None, channel_axis, 0, 1]
        else:
            layer2kernel_axes_map = [None, 0, 1, channel_axis]

        axis = layer2kernel_axes_map[axis]

    broadcast_shape = [1] * weight.ndim
    broadcast_shape[axis] = weight.shape[axis]
    var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
    gamma = np.reshape(gamma, broadcast_shape)
    beta = np.reshape(beta, broadcast_shape)
    bias = np.reshape(bias, broadcast_shape)
    mean = np.reshape(mean, broadcast_shape)
    bias_bn = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)
    weight_bn = weight * gamma * var_eps_sqrt_inv

    return weight_bn, bias_bn


def modify_parameter_precision(weights, biases, config, attributes):
    if config.getboolean('cell', 'binarize_weights'):
        from snntoolbox.utils.utils import binarize
        print("Binarizing weights.")
        weights = binarize(weights)
    elif config.getboolean('cell', 'quantize_weights'):
        assert 'Qm.f' in attributes, \
            "In the [cell] section of the configuration file, " \
            "'quantize_weights' was set to True. For this to " \
            "work, the layer needs to specify the fixed point " \
            "number format 'Qm.f'."
        from snntoolbox.utils.utils import reduce_precision
        m, f = attributes.get('Qm.f')
        print("Quantizing weights to Q{}.{}.".format(m, f))
        weights = reduce_precision(weights, m, f)
        if attributes.get('quantize_bias', False):
            biases = reduce_precision(biases, m, f)

    # These attributes are not needed any longer and would not be
    # understood by Keras when building the parsed model.
    attributes.pop('quantize_bias', None)
    attributes.pop('Qm.f', None)

    return weights, biases


def padding_string(pad, pool_size):
    """Get string defining the border mode.

    Parameters
    ----------

    pad: tuple[int]
        Zero-padding in x- and y-direction.
    pool_size: list[int]
        Size of kernel.

    Returns
    -------

    padding: str
        Border mode identifier.
    """

    if isinstance(pad, str):
        return pad

    if pad == (0, 0):
        padding = 'valid'
    elif pad == (pool_size[0] // 2, pool_size[1] // 2):
        padding = 'same'
    elif pad == (pool_size[0] - 1, pool_size[1] - 1):
        padding = 'full'
    else:
        raise NotImplementedError(
            "Padding {} could not be interpreted as any of the ".format(pad) +
            "supported border modes 'valid', 'same' or 'full'.")
    return padding


def load_parameters(filepath):
    """Load all layer parameters from an HDF5 file."""

    import h5py

    f = h5py.File(filepath, 'r')

    params = []
    for k in sorted(f.keys()):
        params.append(np.array(f.get(k)))

    f.close()

    return params


def save_parameters(params, filepath, fileformat='h5'):
    """Save all layer parameters to an HDF5 file."""

    if fileformat == 'pkl':
        pickle.dump(params, open(filepath + '.pkl', str('wb')))
    else:
        import h5py
        with h5py.File(filepath, mode='w') as f:
            for i, p in enumerate(params):
                if i < 10:
                    j = '00' + str(i)
                elif i < 100:
                    j = '0' + str(i)
                else:
                    j = str(i)
                f.create_dataset('param_' + j, data=p)


def has_weights(layer):
    """Return ``True`` if layer has weights.

    Parameters
    ----------

    layer : keras.layers.Layer
        Keras layer

    Returns
    -------

    : bool
        ``True`` if layer has weights.
    """

    return len(layer.weights)


def get_inbound_layers_with_params(layer):
    """Iterate until inbound layers are found that have parameters.

    Parameters
    ----------

    layer: keras.layers.Layer
        Layer

    Returns
    -------

    : list
        List of inbound layers.
    """

    inbound = layer
    while True:
        inbound = get_inbound_layers(inbound)
        if len(inbound) == 1:
            inbound = inbound[0]
            if has_weights(inbound):
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if has_weights(inb):
                    result.append(inb)
                else:
                    result += get_inbound_layers_with_params(inb)
            return result


def get_inbound_layers_without_params(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    return [layer for layer in get_inbound_layers(layer)
            if not has_weights(layer)]


def get_inbound_layers(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    try:
        # noinspection PyProtectedMember
        inbound_layers = layer._inbound_nodes[0].inbound_layers
    except AttributeError:  # For Keras backward-compatibility.
        inbound_layers = layer.inbound_nodes[0].inbound_layers
    if not isinstance(inbound_layers, (list, tuple)):
        inbound_layers = [inbound_layers]
    return inbound_layers


def get_outbound_layers(layer):
    """Return outbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of outbound layers.
    """

    try:
        # noinspection PyProtectedMember
        outbound_nodes = layer._outbound_nodes
    except AttributeError:  # For Keras backward-compatibility.
        outbound_nodes = layer.outbound_nodes
    return [on.outbound_layer for on in outbound_nodes]


def get_outbound_activation(layer):
    """
    Iterate over 2 outbound layers to find an activation layer. If there is no
    activation layer, take the activation of the current layer.

    Parameters
    ----------

    layer: Union[keras.layers.Conv2D, keras.layers.Dense]
        Layer

    Returns
    -------

    activation: str
        Name of outbound activation type.
    """

    activation = layer.activation.__name__
    outbound = layer
    for _ in range(2):
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1 and get_type(outbound[0]) == 'Activation':
            activation = outbound[0].activation.__name__
    return activation


def get_fanin(layer):
    """
    Return fan-in of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer]
         Layer.

    Returns
    -------

    fanin: int
        Fan-in.

    """

    layer_type = get_type(layer)
    if 'Conv' in layer_type:
        ax = 1 if IS_CHANNELS_FIRST else -1
        fanin = np.prod(layer.kernel_size) * layer.input_shape[ax]
    elif 'Dense' in layer_type:
        fanin = layer.input_shape[1]
    elif 'Pool' in layer_type:
        fanin = 0
    else:
        fanin = 0

    return fanin


def get_fanout(layer, config):
    """
    Return fan-out of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer]
         Layer.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    fanout: Union[int, ndarray]
        Fan-out. The fan-out of a neuron projecting onto a convolution layer
        varies between neurons in a feature map if the stride of the
        convolution layer is greater than unity. In this case, return an array
        of the same shape as the layer.
    """

    from snntoolbox.simulation.utils import get_spiking_outbound_layers

    # In branched architectures like GoogLeNet, we have to consider multiple
    # outbound layers.
    next_layers = get_spiking_outbound_layers(layer, config)
    fanout = 0
    for next_layer in next_layers:
        if 'Conv' in next_layer.name and not has_stride_unity(next_layer):
            shape = layer.output_shape
            if 'Input' in get_type(layer):
                shape = fix_input_layer_shape(shape)
            fanout = np.zeros(shape[1:])
            break

    for next_layer in next_layers:
        if 'Dense' in next_layer.name:
            fanout += next_layer.units
        elif 'Pool' in next_layer.name:
            fanout += 1
        elif 'DepthwiseConv' in next_layer.name:
            if has_stride_unity(next_layer):
                fanout += np.prod(next_layer.kernel_size)
            else:
                fanout += get_fanout_array(layer, next_layer, True)
        elif 'Conv' in next_layer.name:
            if has_stride_unity(next_layer):
                fanout += np.prod(next_layer.kernel_size) * next_layer.filters
            else:
                fanout += get_fanout_array(layer, next_layer)

    return fanout


def has_stride_unity(layer):
    """Return `True` if the strides in all dimensions of a ``layer`` are 1."""

    return all([s == 1 for s in layer.strides])


def get_fanout_array(layer_pre, layer_post, is_depthwise_conv=False):
    """
    Return an array of the same shape as ``layer_pre``, where each entry gives
    the number of outgoing connections of a neuron. In convolution layers where
    the post-synaptic layer has stride > 1, the fan-out varies between neurons.
    """

    ax = 1 if IS_CHANNELS_FIRST else 0

    nx = layer_post.output_shape[2 + ax]  # Width of feature map
    ny = layer_post.output_shape[1 + ax]  # Height of feature map
    nz = layer_post.output_shape[ax]  # Number of channels
    kx, ky = layer_post.kernel_size  # Width and height of kernel
    px = int((kx - 1) / 2) if layer_post.padding == 'same' else 0
    py = int((ky - 1) / 2) if layer_post.padding == 'same' else 0
    sx = layer_post.strides[1]
    sy = layer_post.strides[0]

    shape = layer_pre.output_shape
    if 'Input' in get_type(layer_pre):
        shape = fix_input_layer_shape(shape)
    fanout = np.zeros(shape[1:])

    for y_pre in range(fanout.shape[0 + ax]):
        y_post = [int((y_pre + py) / sy)]
        wy = (y_pre + py) % sy
        i = 1
        while wy + i * sy < ky:
            y = y_post[0] - i
            if 0 <= y < ny:
                y_post.append(y)
            i += 1
        for x_pre in range(fanout.shape[1 + ax]):
            x_post = [int((x_pre + px) / sx)]
            wx = (x_pre + px) % sx
            i = 1
            while wx + i * sx < kx:
                x = x_post[0] - i
                if 0 <= x < nx:
                    x_post.append(x)
                i += 1

            if ax:
                fanout[:, y_pre, x_pre] = len(x_post) * len(y_post)
            else:
                fanout[y_pre, x_pre, :] = len(x_post) * len(y_post)

    if not is_depthwise_conv:
        fanout *= nz

    return fanout


def get_type(layer):
    """Get type of Keras layer.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Keras layer.

    Returns
    -------

    : str
        Layer type.

    """

    return layer.__class__.__name__


def get_quantized_activation_function_from_string(activation_str):
    """
    Parse a string describing the activation of a layer, and return the
    corresponding activation function.

    Parameters
    ----------

    activation_str : str
        Describes activation.

    Returns
    -------

    activation : functools.partial
        Activation function.

    Examples
    --------

    >>> f = get_quantized_activation_function_from_string('relu_Q1.15')
    >>> f
    functools.partial(<function reduce_precision at 0x7f919af92b70>,
                      f='15', m='1')
    >>> print(f.__name__)
    relu_Q1.15
    """

    # TODO: We implicitly assume relu activation function here. Change this to
    #       allow for general activation functions with reduced precision.

    from functools import partial
    from snntoolbox.utils.utils import quantized_relu

    m, f = map(int, activation_str[activation_str.index('_Q') + 2:].split('.'))
    activation = partial(quantized_relu, m=m, f=f)
    activation.__name__ = activation_str

    return activation


def get_clamped_relu_from_string(activation_str):

    from snntoolbox.utils.utils import ClampedReLU

    threshold, max_value = map(eval, activation_str.split('_')[-2:])

    activation = ClampedReLU(threshold, max_value)

    return activation


def get_noisy_softplus_from_string(activation_str):
    from snntoolbox.utils.utils import NoisySoftplus

    k, sigma = map(eval, activation_str.split('_')[-2:])

    activation = NoisySoftplus(k, sigma)

    return activation


def get_custom_activation(activation_str):
    """
    If ``activation_str`` describes a custom activation function, import this
    function from `snntoolbox.utils.utils` and return it. If custom activation
    function is not found or implemented, return the ``activation_str`` in
    place of the activation function.

    Parameters
    ----------

    activation_str : str
        Describes activation.

    Returns
    -------

    activation :
        Activation function.
    activation_str : str
        Describes activation.
    """

    if activation_str == 'binary_sigmoid':
        from snntoolbox.utils.utils import binary_sigmoid
        activation = binary_sigmoid
    elif activation_str == 'binary_tanh':
        from snntoolbox.utils.utils import binary_tanh
        activation = binary_tanh
    elif '_Q' in activation_str:
        activation = get_quantized_activation_function_from_string(
            activation_str)
    elif 'clamped_relu' in activation_str:
        activation = get_clamped_relu_from_string(activation_str)
    elif 'NoisySoftplus' in activation_str:
        from snntoolbox.utils.utils import NoisySoftplus
        activation = NoisySoftplus
    else:
        activation = activation_str

    return activation, activation_str


def assemble_custom_dict(*args):
    assembly = []
    for arg in args:
        assembly += arg.items()
    return dict(assembly)


def get_custom_layers_dict(filepath=None):
    """
    Import all implemented custom layers so they can be used when loading a
    Keras model.

    Parameters
    ----------

    filepath : Optional[str]
        Path to json file containing additional custom objects.
    """

    from snntoolbox.utils.utils import is_module_installed

    custom_layers = {}
    if is_module_installed('keras_rewiring'):
        from keras_rewiring import Sparse, SparseConv2D, SparseDepthwiseConv2D
        from keras_rewiring.optimizers import NoisySGD

        custom_layers.update({'Sparse': Sparse,
                              'SparseConv2D': SparseConv2D,
                              'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
                              'NoisySGD': NoisySGD})

    if filepath is not None and filepath != '':
        with open(filepath) as f:
            kwargs = json.load(f)
            custom_layers.update(kwargs)

    return custom_layers


def get_custom_activations_dict(filepath=None):
    """
    Import all implemented custom activation functions so they can be used when
    loading a Keras model.

    Parameters
    ----------

    filepath : Optional[str]
        Path to json file containing additional custom objects.
    """

    from snntoolbox.utils.utils import binary_sigmoid, binary_tanh, \
        ClampedReLU, LimitedReLU, NoisySoftplus

    # Todo: We should be able to load a different activation for each layer.
    #       Need to remove this hack:
    activation_str = 'relu_Q1.4'
    activation = get_quantized_activation_function_from_string(activation_str)

    custom_objects = {
        'binary_sigmoid': binary_sigmoid,
        'binary_tanh': binary_tanh,
        # Todo: This should work regardless of the specific attributes of the
        #       ClampedReLU class used during training.
        'clamped_relu': ClampedReLU(),
        'LimitedReLU': LimitedReLU,
        'relu6': LimitedReLU({'max_value': 6}),
        activation_str: activation,
        'Noisy_Softplus': NoisySoftplus,
        'precision': precision,
        'activity_regularizer': keras.regularizers.l1}

    if filepath is not None and filepath != '':
        with open(filepath) as f:
            kwargs = json.load(f)

        for key in kwargs:
            if 'LimitedReLU' in key:
                custom_objects[key] = LimitedReLU(kwargs[key])

    return custom_objects


def check_for_custom_activations(layer_attributes):
    """
    Check if the layer contains a custom activation function, and deal with it
    appropriately.

    Parameters
    ----------

    layer_attributes: dict
        A dictionary containing the attributes of the layer.
    """

    if 'activation' not in layer_attributes.keys():
        return


def precision(y_true, y_pred):
    """Precision metric.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant. Only computes a batch-wise average of
    precision.
    """

    import tensorflow.keras.backend as k
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + k.epsilon())


def fix_input_layer_shape(shape):
    """
    tf.keras.models.load_model function introduced a bug that wraps the input
    tensors and shapes in a single-entry list, i.e.
    output_shape == [(None, 1, 28, 28)]. Thus we have to apply [0] here.
    """

    if len(shape) == 1:
        return shape[0]
    return shape
