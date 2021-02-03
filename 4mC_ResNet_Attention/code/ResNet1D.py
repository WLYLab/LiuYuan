from __future__ import division
import six
from keras.layers import *
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    LSTM,
    Bidirectional,
    Dropout
)
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
    )
from keras.layers import add
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras import initializers
from Attention import AttentionLayer
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    #stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    #equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
    equal_cols = input_shape[COL_AXIS] == residual_shape[COL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or not equal_cols:#or stride_height > 1 or not equal_channels
        shortcut = Conv1D(filters=residual_shape[COL_AXIS],
                          kernel_size=1,#(1, 1)
                          strides=stride_width,#, stride_height
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = 1#(1, 1)
            if i == 0 and not is_first_layer:
                init_strides = 2#(2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters=filters, kernel_size=3,#(3, 3)
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=3,#(3, 3)
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=3)(conv1)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    ROW_AXIS = 1
    COL_AXIS = 2
    #CHANNEL_AXIS = 3

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

#=================================================================================
#attention
TIME_STEPS =6
#lstm_units = 1024
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols)")

        # Permute dimension order if necessary
        
        input_shape = (input_shape[0], input_shape[1])#, input_shape[0]

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=7, strides=1)(input)
        #pool1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv1)

        block_1 = conv1
        filters = 64
        for i, r in enumerate(repetitions):
            block_1 = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block_1)
            filters *= 2
        
        # Last activation
        block = _bn_relu(block_1)

        # Classifier block
        block_shape = K.int_shape(block)
        #pool2 = AveragePooling1D(pool_size=(block_shape[ROW_AXIS]),#, block_shape[COL_AXIS]
        #                         strides=1)(block)

        #=========================================================================================================
        #### lstm + attention
        #x = Dropout(0.3)(block)
        #print(x.shape)
        print('drop')
        lstm_units = int(block.shape[-1] // 2)
        print(lstm_units)
        lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(block)
        att=AttentionLayer()(lstm_out)
        #attention_mul = attention_3d_block(lstm_out)
        #attention_flatten = Flatten()(att)
        drop2 = Dropout(0.3)(att)
        output = Dense(2, activation='softmax')(drop2)
        model = Model(inputs=input, outputs=output)

        #==========================================================================================================
        #dense = Dense(units=num_outputs, kernel_initializer="he_normal",activation="softmax")(flatten1)#softmax
        #model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])
if __name__ == "__main__":
    model=ResnetBuilder.build_resnet_18((41, 4), 2)
    model.summary()