import config_tf
import tensorflow as tf
from dataset import CardsWarper, PrioritizedReplay, create_dataset
from models import CardsEmbeding_v5
import gc

wp = CardsWarper( 'cards.json' )

embeding_size = 64
num_blocks = 2
num_heads = 2

max_repeat = 5

with tf.device('/device:GPU:1'):
    
    model = CardsEmbeding_v5( 'test_v5_1', '/media/lucas/DADOS/saved/', len(wp), embeding_size, 0, max_repeat, num_blocks, num_heads )
    # model.load_training( 'saved/', model.m_name )
    
    # train base
    ds_size = 10000
    bs = 512
    epoch = 1000
    hand_size = ( max_repeat // 2 ) * len( wp )

    f_ds = lambda s: create_dataset( wp, s, hand_size, max_repeat )
    dsx, dsy, dsy1 = f_ds( ds_size )

    replay = PrioritizedReplay( ds_size, 1, wp, hand_size )
    [ replay.store( x, y, z ) for x, y, z in zip( dsx, dsy, dsy1 ) ]
    
    model.train_base( replay, f_ds, bs, epoch )

    # train total
    # ds_size = 10000
    # bs = 128
    # epoch = 1000
    # hand_size = 32

    # f_ds = lambda s: create_dataset( wp, s, hand_size, max_repeat )
    # dsx, dsy = f_ds( ds_size )

    # replay = PrioritizedReplay( ds_size, 1, wp, hand_size )
    # [ replay.store( x, y, z ) for x, y, z in zip( dsx, dsy, dsy1 ) ]

    # model.train( replay, f_ds, bs, epoch )

    gc.collect()

