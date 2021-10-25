import config_tf
import tensorflow as tf
from dataset import CardsWarper, PrioritizedReplay, create_dataset
from models import CardsEmbeding

wp = CardsWarper( 'cards.json' )

ds_size = 10000
bs = 128
epoch = 10000

max_repeat = 5
hand_size = ( max_repeat // 2 ) * len( wp )

embeding_size = 32
num_blocks = 4
num_heads = 4

f_ds = lambda s: create_dataset( wp, s, hand_size, max_repeat )
dsx, dsy = f_ds( ds_size )

replay = PrioritizedReplay( ds_size, 1, wp, hand_size )
[ replay.store( x, y ) for x, y in zip( dsx, dsy ) ]

with tf.device('/device:GPU:1'):
    model = CardsEmbeding( 'test6', 'logs', len(wp), embeding_size, hand_size, max_repeat, num_blocks, num_heads )
    model.load_training( 'saved/', model.m_name )
    model.train( replay, f_ds, bs, epoch )

