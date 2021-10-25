import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.activations import gelu
from tensorflow.keras.initializers import RandomNormal

from layers import Conv1D, Nalu, Norm
from ann_utils import attn, positions_for

class TransformerBlock(Layer):

    def __init__(self, num_outputs, n_heads, initializer=RandomNormal(stddev=0.01)):
        
        super(TransformerBlock, self).__init__()

        self.qk = Conv1D( num_outputs * 2, 'qk', initializer = initializer )
        self.v = Nalu( num_outputs, 'v', initializer = initializer )
        # self.v = Dense( num_outputs )
        
        self.ln1 = Conv1D( num_outputs * 4, 'ln1', initializer = initializer )
        self.ln2 = Conv1D( num_outputs, 'ln2', initializer = initializer )
        
        self.merge = Conv1D( num_outputs, 'merge', initializer = initializer )

        self.norm1 = Norm()
        self.norm2 = Norm()
        
        self.n_heads = n_heads

        self.num_outputs = num_outputs
    
    def call(self, input, eval=False):
        
        x = input
        a, msk = attn( self.norm1( x ), self.qk, self.v, self.merge, self.n_heads )
        xa = x + a
        m1 = gelu( self.ln1( self.norm2( xa ) ) )
        m = self.ln2( m1 )
        xm = xa + m

        return xm, msk

class TransformerLayer(Layer):

    def __init__(self, num_outputs, n_heads, num_blocks, max_len, name, pos_emb=False, initializer=RandomNormal(stddev=0.01)):
        
        super(TransformerLayer, self).__init__(name=name)

        self.max_len = max_len
        self.num_outputs = num_outputs
        self.num_blocks = num_blocks
        self.initializer = initializer
        self.pos_emb = pos_emb
        self.blocks = [ TransformerBlock( num_outputs, n_heads ) for _ in range( num_blocks ) ]
    
    def build(self, input_shape):
        
        if self.pos_emb:
            self.wpe = self.add_weight( 'positional_embeding', shape = [ self.max_len, self.num_outputs ], initializer = self.initializer )
        
    def call(self, input, eval=False):

        x = input
        
        if self.pos_emb:
            h = x + tf.gather( self.wpe, positions_for( x, 0 ) )
        else:
            h = x

        msks = []
        for idx in range( len( self.blocks ) ):            
            h, msk = self.blocks[idx]( h, eval )
            msks.append( msk )
        
        msks = tf.stack( msks, axis = 0 )

        return h, msks
