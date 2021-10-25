import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal

from ann_utils import shape_list

class Norm(Layer):

    def __init__(self, axis=-1, epsilon=1e-5):
        
        super(Norm, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):

        self.g = self.add_weight( 'g', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer( 1 ), trainable = True )
        self.b = self.add_weight( 'b', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer( 0 ), trainable = True )
        
    def call(self, input):

        x = input
        u = tf.reduce_mean( x, axis = self.axis, keepdims = True )
        s = tf.reduce_mean( tf.square( x - u ), axis = self.axis, keepdims = True )
        nx = ( x - u ) * tf.math.rsqrt( s + self.epsilon )
        nx = ( nx * self.g ) + self.b
        
        return nx

class Conv1D(Layer):

    def __init__(self, nf, name, initializer=RandomNormal()):
        super(Conv1D, self).__init__()  
        self.initializer = initializer
        self.nf = nf
        self.nme = name

    def build(self, input_shape):

        self.kernel = self.add_weight( '{}_kernel_1d'.format( self.nme ), shape = [ 1, input_shape[ -1 ], self.nf ], initializer = self.initializer )
        self.bias = self.add_weight( '{}_bias'.format( self.nme ), shape = [ self.nf ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):
        *start, nx = shape_list( input )
        i = tf.reshape( input, [ -1, nx ] )
        w = tf.reshape( self.kernel, [ -1, self.nf ] )
        c = tf.matmul( i, w )
        cb = c + self.bias
        r = tf.reshape( cb, start + [ self.nf ] )
        return r

class Nalu(Layer):

    def __init__(self, num_outputs, name, initializer=TruncatedNormal()):
        
        super(Nalu, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.initializer = initializer
    
    def build(self, input_shape):

        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

    def call(self, input):

        x1 = input

        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        g = tf.sigmoid( tf.matmul( x1, self.gt ) )
        a = tf.matmul( x1, w )
        m = tf.sinh( tf.matmul( tf.asinh( x1 ), w ) )
        arithimetic_x = ( g * a ) + ( ( 1 - g ) * m )

        return arithimetic_x

class Adam(tf.keras.optimizers.Adam):
    
    def __init__(self, learning_rate, beta_1 = tf.Variable(0.9), beta_2 = tf.Variable(0.999), epsilon = tf.Variable(1e-7), decay = tf.Variable(0.0)):
        super(Adam, self).__init__( learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon )
        self.iterations
        self.decay = decay

class RMS(tf.keras.optimizers.RMSprop):
    
    def __init__(self, learning_rate, rho=tf.Variable(0.9), momentum=tf.Variable(0.0), epsilon=tf.Variable(1e-07), centered=False):
        super(RMS, self).__init__( learning_rate, rho=rho, momentum=momentum, epsilon=epsilon, centered=centered )
        self.iterations
        self.decay = tf.Variable(0.0)

class AdamW(tfa.optimizers.AdamW):
    
    def __init__(self, weight_decay, learning_rate, beta_1 = tf.Variable(0.9), beta_2 = tf.Variable(0.999), epsilon = tf.Variable(1e-7), decay = tf.Variable(0.0)):
        super(AdamW, self).__init__( weight_decay = weight_decay, learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, amsgrad = False )
        self.iterations
        self.decay = decay

