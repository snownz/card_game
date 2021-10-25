import config_tf
from dataset import CardsWarper
from models import CardsEmbeding
import numpy as np
import tensorflow as tf

hand_size = 6
max_repeat = 5
embeding_size = 32
num_blocks = 4
num_heads = 4

wp = CardsWarper( 'cards.json' )

with tf.device('/device:CPU:0'):

    model = CardsEmbeding( 'test6', 'logs', len(wp), embeding_size, hand_size, max_repeat, num_blocks, num_heads )
    model.load_training( 'saved/', model.m_name )

    dec1 = ( np.random.uniform( 0, 1, [ 1, hand_size ] ) * ( len( wp ) ) ).astype( np.int32 )
    dec2 = ( np.random.uniform( 0, 1, [ 1, hand_size ] ) * ( len( wp ) ) ).astype( np.int32 )
    dec3 = ( np.random.uniform( 0, 1, [ 1, hand_size ] ) * ( len( wp ) ) ).astype( np.int32 )

    o1, m1 = model( dec1 )
    o2, m2 = model( dec2 )
    o3, m3 = model( dec3 )

    e1 = model.get_embeding( dec1 )
    e2 = model.get_embeding( dec2 )
    e3 = model.get_embeding( dec3 )

    # op = e1 - e2
    op = e1 + e2
    # op = ( e1 + e2 ) - e3
    # op = tf.maximum( e1, e2 )
    # op = tf.minimum( e1, e2 )

    e3, e3m = model.reconstruct( op )

    a = 10
