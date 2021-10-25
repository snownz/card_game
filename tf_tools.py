import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec


def save_checkpoint(model, base_dir, step):

    checkpoint = tf.train.Checkpoint( model = model )
    manager = tf.train.CheckpointManager( checkpoint, directory = base_dir, max_to_keep = 5 )
    manager.save( checkpoint_number = step )

def restore_checkpoint(model, base_dir):
    
    checkpoint = tf.train.Checkpoint( model = model )
    manager = tf.train.CheckpointManager( checkpoint, directory = base_dir, max_to_keep = 5 )

    print("\n\n=========================================================================================\n")
    print("Loading from: {} ckp - {}".format(base_dir, int( manager.latest_checkpoint.split('-')[-1] )))
    print("\n=========================================================================================\n\n")

    status = checkpoint.restore( manager.latest_checkpoint ).expect_partial()
    status.assert_existing_objects_matched()
    status.assert_consumed()
    return int( manager.latest_checkpoint.split('-')[-1] )


class Module(tf.Module):

  def __init__(self):

    self.t_step = 0

  def save_tarining(self, folder, name):
    save_checkpoint( self, folder + name, self.t_step )

  def load_training(self, folder, name):
    self.t_step = restore_checkpoint( self, folder + name )

  def save_graph(self, folder, name):
    dir = folder + '/' + name
    if not os.path.isdir( dir ):
      os.mkdir( dir )
    num = len( [ x for x in os.listdir( dir ) if os.path.isdir( dir + '/' + x ) ] )
    tf.saved_model.save( self, dir + '/' + str( num ) )

