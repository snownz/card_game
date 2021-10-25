import tensorflow as tf
from functools import partial


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    ex = tf.exp(x)
    return ex / tf.reduce_sum( ex, axis = axis, keepdims = True )

def l2_loss(variables, factor):
    ls = []
    for v in variables:
        ls.append( tf.nn.l2_loss( v ) )
    return factor * tf.reduce_mean( ls )

def expand_tile(value, size):

    """Add a new axis of given size."""
    value = tf.convert_to_tensor( value )
    ndims = value.shape.ndims
    return tf.tile( tf.expand_dims( value, axis = 0 ), [ size ] + [ 1 ] * ndims )

def positions_for(sequences, past_length):

    batch_size = tf.shape( sequences )[0]
    nsteps = tf.shape( sequences )[1]
    return expand_tile( past_length + tf.range( nsteps ), batch_size )

def attn(features, qk, v, merge, n_head):

    def apply_mask(x, mask):
        return x * mask - tf.cast( 1e10, x.dtype ) * ( 1 - mask )
        
    def attention_mask(nd, ns, dtype):
        i = tf.range( nd )[:,None]
        j = tf.range( ns )
        m = i >= j - ns + nd
        return tf.cast( m, dtype )

    def merge_states(x):
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a*b])

    def split_states(x, n):
        *start, m = shape_list(x)
        return tf.reshape(x, start + [n, m//n])

    def split_heads(x, n_head):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list( w )

        b = attention_mask( nd, ns, dtype = w.dtype )
        b = tf.reshape( b, [ 1, 1, nd, ns ] )

        w = apply_mask( w, b )

        return w, b

    def multihead_attn(q, k, v):

        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )
        w, b = mask_attn_weights( w )
        w = tf.nn.softmax( w )
        a = tf.matmul( w, v )
        return a, w
    
    c1 = qk( features )
    c2 = v( features )
    q, k, v = map( partial( split_heads, n_head = n_head ), tf.split( c1, 2, axis = 2 ) + [ c2 ] )
    a, msk = multihead_attn( q, k, v )
    ah = merge_heads( a )        
    am = merge( ah )

    return am, msk
