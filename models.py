import tensorflow as tf
from tf_tools import Module
from networks import TransformerLayer
from layers import Adam, RMS, AdamW, Nalu
from ann_utils import shape_list

from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax, gelu, relu, tanh, softsign, sigmoid
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy, mean_squared_error
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from tqdm import tqdm
from random import randint

import numpy as np

class CardsEmbeding_v5( Module ):

    def __init__(self, name, log_dir, num_cards, embeding_size, hand_size, max_repeat, num_blocks, num_heads):
    
        super(CardsEmbeding_v5, self).__init__()

        self.module_type = 'CardsEmbeding_v5'
        self.m_name = name

        self.log_dir = log_dir + '/' + name

        self.num_cards = num_cards
        self.embeding_size = embeding_size
        self.max_repeat = max_repeat

        # model
        # embeding base
        self.embeding = tf.Variable( tf.random.normal( [ num_cards, embeding_size ], 0, 0.02 ), trainable = True, name = 'base_cards_embeding' )
        self.layer = TransformerLayer( embeding_size, num_heads, num_blocks, hand_size, 'base_tlayer' )
        self.l1 = Dense( embeding_size, name = 'base_anchor' )
        
        # base embeding tasks
        self.isin = Dense( embeding_size, name = 'base_h_isin', activation = tanh )
        self.frequency = Nalu( embeding_size, name = 'base_h_frequency' )

        # embeding isin tasks
        self.sub_m = Dense( embeding_size * 2, name = 'h_sub_m', activation = sigmoid )
        self.add_m = Dense( embeding_size * 2, name = 'h_add_m', activation = sigmoid )
        self.mul_m = Dense( embeding_size * 2, name = 'h_mul_m', activation = sigmoid )

        self.add = Dense( embeding_size, name = 'h_add', activation = tanh )
        self.mul = Dense( embeding_size, name = 'h_mul', activation = tanh )
        self.sub = Dense( embeding_size, name = 'h_sub', activation = tanh )

        # embeding count tasks
        # self.sub_c_m = Dense( embeding_size * 2, name = 'h_sub_c_m', activation = sigmoid )
        # self.add_c_m = Dense( embeding_size * 2, name = 'h_add_c_m', activation = sigmoid )
        # self.mul_c_m = Dense( embeding_size * 2, name = 'h_mul_c_m', activation = sigmoid )
        # self.div_c_m = Dense( embeding_size * 2, name = 'h_mul_c_m', activation = sigmoid )

        # self.add_c = Nalu( embeding_size, name = 'h_add_c' )
        # self.mul_c = Nalu( embeding_size, name = 'h_mul_c' )
        # self.sub_c = Nalu( embeding_size, name = 'h_sub_c' )
        # self.div_c = Nalu( embeding_size, name = 'h_sub_c' )
        
        # output sets
        self.out_isin = Dense( num_cards * 2, name = 'base_o_isin' )
        self.out_frequency = Nalu( num_cards, name = 'base_o_frequency' )

        # train variables        
        self.step = tf.Variable( 0, trainable = False )
        boundaries = [ 10000, 20000, 100000 ]
        values = [ 1e-0, 1e-1, 1e-2, 1e-4 ]
        self.schedule = PiecewiseConstantDecay( boundaries, values )
        lr = 2e-3 * self.schedule( self.step )
        wd = lambda: 1e-4 * self.schedule( self.step )

        self.opt = AdamW( wd, lr )

        self.train_loss_count = tf.keras.metrics.Mean( 'train_loss_count', dtype = tf.float32 )

        self.train_loss_isin = tf.keras.metrics.Mean( 'train_loss_isin', dtype = tf.float32 )
        self.train_acc_isin = tf.keras.metrics.Accuracy( 'train_acc_isin', dtype = tf.float32 )

        self.train_loss_tu = tf.keras.metrics.Mean( 'train_loss_u', dtype = tf.float32 )
        self.train_acc_tu = tf.keras.metrics.Accuracy( 'train_acc_u', dtype = tf.float32 )

        self.train_loss_ti = tf.keras.metrics.Mean( 'train_loss_i', dtype = tf.float32 )
        self.train_acc_ti = tf.keras.metrics.Accuracy( 'train_acc_i', dtype = tf.float32 )

        self.train_loss_td = tf.keras.metrics.Mean( 'train_loss_d', dtype = tf.float32 )
        self.train_acc_td = tf.keras.metrics.Accuracy( 'train_acc_d', dtype = tf.float32 )

        self.train_loss_t1 = tf.keras.metrics.Mean( 'train_loss_1', dtype = tf.float32 )
        self.train_acc_t1 = tf.keras.metrics.Accuracy( 'train_acc_1', dtype = tf.float32 )

        self.train_loss_t2 = tf.keras.metrics.Mean( 'train_loss_2', dtype = tf.float32 )
        self.train_acc_t2 = tf.keras.metrics.Accuracy( 'train_acc_2', dtype = tf.float32 )

        self.train_loss_t3 = tf.keras.metrics.Mean( 'train_loss_3', dtype = tf.float32 )
        self.train_acc_t3 = tf.keras.metrics.Accuracy( 'train_acc_3', dtype = tf.float32 )

        self.train_loss_t4 = tf.keras.metrics.Mean( 'train_loss_4', dtype = tf.float32 )
        self.train_acc_t4 = tf.keras.metrics.Accuracy( 'train_acc_4', dtype = tf.float32 )

        self.train_summary_writer_base = tf.summary.create_file_writer( self.log_dir + '/base' )
        self.train_summary_writer_total = tf.summary.create_file_writer( self.log_dir + '/total' )

        self.t_step = 0
        self.t_step_b = 0
        self.t_step_t = 0
        
        in1 = tf.zeros( [ 1, hand_size ], dtype = tf.int32 )
        self.__load__( in1 )

    def __load__(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        
        # anchor
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, _ = self.layer( emb, eval )

        # count cards
        h_isin = self.isin( h )
        h_freq = self.frequency( h )

        self.out_isin( h_isin )
        self.out_frequency( h_freq )

        # sets operations
        # h_sets = self.sets( h )
        h_sets = h_isin

        self.sub_m( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.add_m( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.mul_m( tf.concat( [ h_sets, h_sets ], -1 ) )

        self.sub( h_sets )
        self.add( h_sets )
        self.mul( h_sets )

    def isin_op(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, msk = self.layer( emb, eval )

        # isin cards
        h_isin = self.isin( h )
        h1 = self.out_isin( h_isin )
        h1 = tf.reshape( h1, [ -1, input.shape[1], self.num_cards, 2 ] )
        out_isin = softmax( h1 )

        h_freq = self.frequency( h )
        freq = relu( self.out_frequency( h_freq ) )

        return out_isin, ac, msk, freq

    def get_embeding(self, input, eval=False):
        emb = tf.nn.embedding_lookup( self.embeding, input )
        h, _ = self.layer( emb, eval )
        return self.isin( h ) #, self.sets( h )

    def add_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.add_m( vl ), 2, axis = -1 )
        return self.add( m1 * a + m2 * b )

    def sub_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.sub_m( vl ), 2, axis = -1 )
        return self.sub( m1 * a + m2 * b )

    def mul_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.mul_m( vl ), 2, axis = -1 )
        return self.mul( m1 * a + m2 * b )

    def reconstruct_sets(self, embeding, eval=False):
        h = self.out_isin( embeding )
        h = tf.reshape( h, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h )
        return out_isin

    def rloss(self, x, y):

        loss = mean_squared_error( y, x )
        loss = tf.reduce_mean( loss, -1 )

        return loss

    def loss(self, x, y, w):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )# * w

        return loss

    def loss2(self, x, y):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )

        return loss

    def train_base(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot, y_freq):

            out_isin, ac, msk, freq = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w )
            floss = tf.reduce_mean( self.rloss( freq, y_freq ) )
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) ) + floss

            return ls, tloss, out_isin, msk, tf.reduce_mean(floss)

        vars = [ x for x in self.trainable_variables if 'base' in x.name ]
        for _ in range(epochs):

            dsx, dsy, dsy1 = new_ds( 1000 )
            [ ds.store( x, y, z ) for x, y, z in zip( dsx, dsy, dsy1 ) ]

            for d in tqdm( range( 200 ) ):
       
                x, y, y1, idx, w = ds.sample_batch( bs )

                y_hot = tf.one_hot( y, 2 )
                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk, rloss = main_loss( x, y_hot, y1 )
                    
                    # final loss
                    floss = m_floss

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss_count( rloss )
                self.train_loss_isin( m_floss )
                self.train_acc_isin.update_state( y, tf.argmax( out, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_base.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_b )

                    tf.summary.scalar( 'total/loss_count', self.train_loss_count.result(), step = self.t_step_b )
                    tf.summary.scalar( 'total/loss_isin', self.train_loss_isin.result(), step = self.t_step_b )
                    tf.summary.scalar( 'total/acc_isin', self.train_acc_isin.result(), step = self.t_step_b )
                    tf.summary.scalar( 'total/lr', self.schedule(self.step) * 2e-3, step = self.t_step_b )
                    
                    tf.summary.histogram( 'total/priorities', ds.priorities, step = self.t_step_b )

                    for i, b in enumerate(msk):
                        m = b[0]
                        for j, h in enumerate(m):
                            tf.summary.image( 'train_mosaic/b{}h{}'.format(i,j), 
                                                h[tf.newaxis,:,:,tf.newaxis], step = self.t_step_b, max_outputs = 1 )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_b += 1
                self.step = self.step + 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() )

    def train(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, ac, msk = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        def other_tasks_loss_v1(xs, y1_hot, y2_hot, y3_hot, y4_hot):
            
            xa, xb, xc = tf.split( xs, 3, axis = 0 )
            xa, xb, xc = tf.squeeze( xa, 0 ), tf.squeeze( xb, 0 ), tf.squeeze( xc, 0 )

            # get embedings
            out_a = self.get_embeding( xa )
            out_b = self.get_embeding( xb )
            out_c = self.get_embeding( xc )

            # set operations
            t1  = self.add_op( self.sub_op( out_a, out_b ), out_c )
            t1a = self.add_op( out_c, self.sub_op( out_a, out_b ) )
            t2  = self.sub_op( self.add_op( out_a, out_b ), out_c )
            t3  = self.sub_op( out_c, self.add_op( out_a, out_b ) )
            t4  = self.sub_op( out_c, self.sub_op( out_a, out_b ) )

            # reconstruct, but without gradients
            pred_1  = self.reconstruct_sets( t1 )
            pred_1a = self.reconstruct_sets( t1a )
            pred_2  = self.reconstruct_sets( t2 )
            pred_3  = self.reconstruct_sets( t3 )
            pred_4  = self.reconstruct_sets( t4 )
            
            # compute operation looses
            mloss_1  = tf.reduce_mean( self.loss2( pred_1,  y1_hot ) )
            mloss_1a = tf.reduce_mean( self.loss2( pred_1a, y1_hot ) )
            mloss_2  = tf.reduce_mean( self.loss2( pred_2,  y2_hot ) )
            mloss_3  = tf.reduce_mean( self.loss2( pred_3,  y3_hot ) )
            mloss_4  = tf.reduce_mean( self.loss2( pred_4,  y4_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( ( out_a + out_b + out_c ) / 3.0 )

            pr1  = tf.reduce_mean( t1  )
            pr1a = tf.reduce_mean( t1a )
            pr2  = tf.reduce_mean( t2  )
            pr3  = tf.reduce_mean( t3  )
            pr4  = tf.reduce_mean( t4  )

            rloss_1  = ( yr - pr1  ) ** 2
            rloss_1a = ( yr - pr1a ) ** 2
            rloss_2  = ( yr - pr2  ) ** 2
            rloss_3  = ( yr - pr3  ) ** 2
            rloss_4  = ( yr - pr4  ) ** 2

            # sum all looses
            s_floss = ( mloss_1 + mloss_1a + mloss_2 + mloss_3 + mloss_4 ) + 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )
            # s_floss = 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )

            return s_floss, mloss_1 + mloss_1a, mloss_2, mloss_3, mloss_4,\
                   pred_1, pred_2, pred_3, pred_4

        def other_tasks_loss_v2( xs, yu_hot, yd_hot, ym_hot):

            # get embedings
            _, sz, _ = shape_list( xs )
            xs_r = [ tf.squeeze( x, axis = 1 ) for x in tf.split( xs, sz, axis = 1 ) ]
            out_a = [ self.get_embeding( x ) for x in xs_r ]

            # union
            t_u = out_a[0]
            for x in out_a[1:]: t_u = self.add_op( t_u, x )

            # diference
            t_d = out_a[0]
            for x in out_a[1:]: t_d = self.sub_op( t_d, x )

            # intersection
            t_m = out_a[0]
            for x in out_a[1:]: t_m = self.mul_op( t_m, x )

            # reconstruct, but without gradients
            pred_u = self.reconstruct_sets( t_u )
            pred_d = self.reconstruct_sets( t_d )
            pred_m = self.reconstruct_sets( t_m )
            
            # compute operation looses
            mloss_u = tf.reduce_mean( self.loss2( pred_u, yu_hot ) )
            mloss_d = tf.reduce_mean( self.loss2( pred_d, yd_hot ) )
            mloss_m = tf.reduce_mean( self.loss2( pred_m, ym_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( out_a )

            pru = tf.reduce_mean( t_u )
            prd = tf.reduce_mean( t_d )
            prm = tf.reduce_mean( t_m )

            rloss_u = ( yr - pru ) ** 2
            rloss_d = ( yr - prd ) ** 2
            rloss_m = ( yr - prm ) ** 2

            # sum all looses
            s_floss = ( mloss_u + mloss_d + mloss_m ) + 0.2 * ( rloss_u + rloss_d + rloss_m )
            # s_floss = mloss_d + 0.2 * ( rloss_u + rloss_d + rloss_m )

            return s_floss, mloss_u, mloss_d, mloss_m, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m

        vars = self.trainable_variables
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):

                sz = randint( 2, 10 )                
                x, y, xs1, yu, yd, ym,\
                xs2, y1, y2, y3, y4,\
                idx, w = ds.sample_batch_v2( bs, sz )

                y_hot = tf.one_hot( y, self.max_repeat )
                yu_hot = tf.one_hot( yu, self.max_repeat )
                yd_hot = tf.one_hot( yd, self.max_repeat )
                ym_hot = tf.one_hot( ym, self.max_repeat )

                y1_hot = tf.one_hot( y1, self.max_repeat )
                y2_hot = tf.one_hot( y2, self.max_repeat )
                y3_hot = tf.one_hot( y3, self.max_repeat )
                y4_hot = tf.one_hot( y4, self.max_repeat )

                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # other tasks
                    s_floss, mloss_u, mloss_d, mloss_i, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m =\
                        other_tasks_loss_v2( xs1, yu_hot, yd_hot, ym_hot )

                    # other tasks
                    s_floss1, mloss_1, mloss_2, mloss_3, mloss_4,\
                    pred_1, pred_2, pred_3, pred_4 = other_tasks_loss_v1( xs2, y1_hot, y2_hot, y3_hot, y4_hot )

                    # final loss
                    floss = m_floss + s_floss + s_floss1

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                self.train_loss_tu( mloss_u )
                self.train_acc_tu.update_state( yu, tf.argmax( pred_u, axis = -1 ) )

                self.train_loss_ti( mloss_i )
                self.train_acc_ti.update_state( ym, tf.argmax( pred_m, axis = -1 ) )

                self.train_loss_td( mloss_d )
                self.train_acc_td.update_state( yd, tf.argmax( pred_d, axis = -1 ) )

                self.train_loss_t1( mloss_1 )
                self.train_acc_t1.update_state( y1, tf.argmax( pred_1, axis = -1 ) )

                self.train_loss_t2( mloss_2 )
                self.train_acc_t2.update_state( y2, tf.argmax( pred_2, axis = -1 ) )

                self.train_loss_t3( mloss_3 )
                self.train_acc_t3.update_state( y3, tf.argmax( pred_3, axis = -1 ) )

                self.train_loss_t4( mloss_4 )
                self.train_acc_t4.update_state( y4, tf.argmax( pred_4, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_total.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_t )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/u', self.train_loss_tu.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/u', self.train_acc_tu.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/i', self.train_loss_ti.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/i', self.train_acc_ti.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/d', self.train_loss_td.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/d', self.train_acc_td.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/1', self.train_loss_t1.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/1', self.train_acc_t1.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/2', self.train_loss_t2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/2', self.train_acc_t2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/3', self.train_loss_t3.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/3', self.train_acc_t3.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/4', self.train_loss_t4.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/4', self.train_acc_t4.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary/loss_u_r', rloss_u, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_i_r', rloss_m, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_d_r', rloss_d, step = self.t_step_t )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_t += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() + s_floss.numpy() + s_floss1.numpy() )

class CardsEmbeding_v4( Module ):

    def __init__(self, name, log_dir, num_cards, embeding_size, hand_size, max_repeat, num_blocks, num_heads):
    
        super(CardsEmbeding_v4, self).__init__()

        self.module_type = 'CardsEmbeding_v4'
        self.m_name = name
        max_repeat = 2

        self.log_dir = log_dir + '/' + name

        self.num_cards = num_cards
        self.embeding_size = embeding_size
        self.max_repeat = max_repeat

        # model
        # embeding base
        self.embeding = tf.Variable( tf.random.normal( [ num_cards, embeding_size ], 0, 0.02 ), trainable = True, name = 'base_cards_embeding' )
        self.layer = TransformerLayer( embeding_size, num_heads, num_blocks, hand_size, 'base_tlayer' )
        self.l1 = Dense( embeding_size, name = 'base_anchor' )
        
        # base embeding tasks
        self.isin = Dense( embeding_size, name = 'base_h_isin', activation = tanh )

        # embeding tasks
        self.sub_m = Dense( embeding_size*2, name = 'h_sub_m', activation = sigmoid )
        self.add_m = Dense( embeding_size*2, name = 'h_add_m', activation = sigmoid )
        self.mul_m = Dense( embeding_size*2, name = 'h_mul_m', activation = sigmoid )

        self.add = Dense( embeding_size, name = 'h_add', activation = tanh )
        self.mul = Dense( embeding_size, name = 'h_mul', activation = tanh )
        self.sub = Dense( embeding_size, name = 'h_sub', activation = tanh )
        
        # output sets
        self.out_isin = Dense( num_cards * max_repeat, name = 'base_o_isin' )

        # train variables        
        step = tf.Variable( 0, trainable = False )
        boundaries = [ 100000, 150000 ]
        values = [ 1e-0, 1e-1, 1e-2 ]
        schedule = PiecewiseConstantDecay( boundaries, values )
        lr = 2e-3 * schedule( step )
        wd = lambda: 1e-4 * schedule( step )

        self.opt = AdamW( wd, lr )

        self.train_loss2 = tf.keras.metrics.Mean( 'train_loss2', dtype = tf.float32 )
        self.train_acc2 = tf.keras.metrics.Accuracy( 'train_acc2', dtype = tf.float32 )

        self.train_loss_tu = tf.keras.metrics.Mean( 'train_loss_u', dtype = tf.float32 )
        self.train_acc_tu = tf.keras.metrics.Accuracy( 'train_acc_u', dtype = tf.float32 )

        self.train_loss_ti = tf.keras.metrics.Mean( 'train_loss_i', dtype = tf.float32 )
        self.train_acc_ti = tf.keras.metrics.Accuracy( 'train_acc_i', dtype = tf.float32 )

        self.train_loss_td = tf.keras.metrics.Mean( 'train_loss_d', dtype = tf.float32 )
        self.train_acc_td = tf.keras.metrics.Accuracy( 'train_acc_d', dtype = tf.float32 )

        self.train_loss_t1 = tf.keras.metrics.Mean( 'train_loss_1', dtype = tf.float32 )
        self.train_acc_t1 = tf.keras.metrics.Accuracy( 'train_acc_1', dtype = tf.float32 )

        self.train_loss_t2 = tf.keras.metrics.Mean( 'train_loss_2', dtype = tf.float32 )
        self.train_acc_t2 = tf.keras.metrics.Accuracy( 'train_acc_2', dtype = tf.float32 )

        self.train_loss_t3 = tf.keras.metrics.Mean( 'train_loss_3', dtype = tf.float32 )
        self.train_acc_t3 = tf.keras.metrics.Accuracy( 'train_acc_3', dtype = tf.float32 )

        self.train_loss_t4 = tf.keras.metrics.Mean( 'train_loss_4', dtype = tf.float32 )
        self.train_acc_t4 = tf.keras.metrics.Accuracy( 'train_acc_4', dtype = tf.float32 )

        self.train_summary_writer_base = tf.summary.create_file_writer( self.log_dir + '/base' )
        self.train_summary_writer_total = tf.summary.create_file_writer( self.log_dir + '/total' )

        self.t_step = 0
        self.t_step_b = 0
        self.t_step_t = 0
        
        in1 = tf.zeros( [ 1, hand_size ], dtype = tf.int32 )
        self.__load__( in1 )

    def __load__(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        
        # anchor
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, _ = self.layer( emb, eval )

        # count cards
        h_isin = self.isin( h )
        self.out_isin( h_isin )

        # sets operations
        # h_sets = self.sets( h )
        h_sets = h_isin

        self.sub_m( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.add_m( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.mul_m( tf.concat( [ h_sets, h_sets ], -1 ) )

        self.sub( h_sets )
        self.add( h_sets )
        self.mul( h_sets )

    def isin_op(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, msk = self.layer( emb, eval )

        # isin cards
        h_isin = self.isin( h )
        h1 = self.out_isin( h_isin )
        h1 = tf.reshape( h1, [ -1, input.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h1 )

        return out_isin, ac, msk

    def get_embeding(self, input, eval=False):
        emb = tf.nn.embedding_lookup( self.embeding, input )
        h, _ = self.layer( emb, eval )
        return self.isin( h ) #, self.sets( h )

    def add_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.add_m( vl ), 2, axis = -1 )
        return self.add( m1 * a + m2 * b )

    def sub_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.sub_m( vl ), 2, axis = -1 )
        return self.sub( m1 * a + m2 * b )

    def mul_op(self, a, b):
        vl = tf.concat( [ a, b ], -1 )
        m1, m2 = tf.split( self.mul_m( vl ), 2, axis = -1 )
        return self.mul( m1 * a + m2 * b )

    def reconstruct_sets(self, embeding, eval=False):
        h = self.out_isin( embeding )
        h = tf.reshape( h, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h )
        return out_isin

    def loss(self, x, y, w):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )# * w

        return loss

    def loss2(self, x, y):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )

        return loss

    def train_base(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, ac, msk = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        vars = [ x for x in self.trainable_variables if 'base' in x.name ]
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):
       
                x, y, idx, w = ds.sample_batch( bs )

                y_hot = tf.one_hot( y, self.max_repeat )
                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # final loss
                    floss = m_floss

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_base.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_b )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step_b )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step_b )

                    tf.summary.histogram( 'total/priorities', ds.priorities, step = self.t_step_b )

                    for i, b in enumerate(msk):
                        m = b[0]
                        for j, h in enumerate(m):
                            tf.summary.image( 'train_mosaic/b{}h{}'.format(i,j), 
                                                h[tf.newaxis,:,:,tf.newaxis], step = self.t_step_b, max_outputs = 1 )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_b += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() )

    def train(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, ac, msk = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        def other_tasks_loss_v1(xs, y1_hot, y2_hot, y3_hot, y4_hot):
            
            xa, xb, xc = tf.split( xs, 3, axis = 0 )
            xa, xb, xc = tf.squeeze( xa, 0 ), tf.squeeze( xb, 0 ), tf.squeeze( xc, 0 )

            # get embedings
            out_a = self.get_embeding( xa )
            out_b = self.get_embeding( xb )
            out_c = self.get_embeding( xc )

            # set operations
            t1  = self.add_op( self.sub_op( out_a, out_b ), out_c )
            t1a = self.add_op( out_c, self.sub_op( out_a, out_b ) )
            t2  = self.sub_op( self.add_op( out_a, out_b ), out_c )
            t3  = self.sub_op( out_c, self.add_op( out_a, out_b ) )
            t4  = self.sub_op( out_c, self.sub_op( out_a, out_b ) )

            # reconstruct, but without gradients
            pred_1  = self.reconstruct_sets( t1 )
            pred_1a = self.reconstruct_sets( t1a )
            pred_2  = self.reconstruct_sets( t2 )
            pred_3  = self.reconstruct_sets( t3 )
            pred_4  = self.reconstruct_sets( t4 )
            
            # compute operation looses
            mloss_1  = tf.reduce_mean( self.loss2( pred_1,  y1_hot ) )
            mloss_1a = tf.reduce_mean( self.loss2( pred_1a, y1_hot ) )
            mloss_2  = tf.reduce_mean( self.loss2( pred_2,  y2_hot ) )
            mloss_3  = tf.reduce_mean( self.loss2( pred_3,  y3_hot ) )
            mloss_4  = tf.reduce_mean( self.loss2( pred_4,  y4_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( ( out_a + out_b + out_c ) / 3.0 )

            pr1  = tf.reduce_mean( t1  )
            pr1a = tf.reduce_mean( t1a )
            pr2  = tf.reduce_mean( t2  )
            pr3  = tf.reduce_mean( t3  )
            pr4  = tf.reduce_mean( t4  )

            rloss_1  = ( yr - pr1  ) ** 2
            rloss_1a = ( yr - pr1a ) ** 2
            rloss_2  = ( yr - pr2  ) ** 2
            rloss_3  = ( yr - pr3  ) ** 2
            rloss_4  = ( yr - pr4  ) ** 2

            # sum all looses
            s_floss = ( mloss_1 + mloss_1a + mloss_2 + mloss_3 + mloss_4 ) + 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )
            # s_floss = 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )

            return s_floss, mloss_1 + mloss_1a, mloss_2, mloss_3, mloss_4,\
                   pred_1, pred_2, pred_3, pred_4

        def other_tasks_loss_v2( xs, yu_hot, yd_hot, ym_hot):

            # get embedings
            _, sz, _ = shape_list( xs )
            xs_r = [ tf.squeeze( x, axis = 1 ) for x in tf.split( xs, sz, axis = 1 ) ]
            out_a = [ self.get_embeding( x ) for x in xs_r ]

            # union
            t_u = out_a[0]
            for x in out_a[1:]: t_u = self.add_op( t_u, x )

            # diference
            t_d = out_a[0]
            for x in out_a[1:]: t_d = self.sub_op( t_d, x )

            # intersection
            t_m = out_a[0]
            for x in out_a[1:]: t_m = self.mul_op( t_m, x )

            # reconstruct, but without gradients
            pred_u = self.reconstruct_sets( t_u )
            pred_d = self.reconstruct_sets( t_d )
            pred_m = self.reconstruct_sets( t_m )
            
            # compute operation looses
            mloss_u = tf.reduce_mean( self.loss2( pred_u, yu_hot ) )
            mloss_d = tf.reduce_mean( self.loss2( pred_d, yd_hot ) )
            mloss_m = tf.reduce_mean( self.loss2( pred_m, ym_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( out_a )

            pru = tf.reduce_mean( t_u )
            prd = tf.reduce_mean( t_d )
            prm = tf.reduce_mean( t_m )

            rloss_u = ( yr - pru ) ** 2
            rloss_d = ( yr - prd ) ** 2
            rloss_m = ( yr - prm ) ** 2

            # sum all looses
            s_floss = ( mloss_u + mloss_d + mloss_m ) + 0.2 * ( rloss_u + rloss_d + rloss_m )
            # s_floss = mloss_d + 0.2 * ( rloss_u + rloss_d + rloss_m )

            return s_floss, mloss_u, mloss_d, mloss_m, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m

        vars = self.trainable_variables
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):

                sz = randint( 2, 10 )                
                x, y, xs1, yu, yd, ym,\
                xs2, y1, y2, y3, y4,\
                idx, w = ds.sample_batch_v2( bs, sz )

                y_hot = tf.one_hot( y, self.max_repeat )
                yu_hot = tf.one_hot( yu, self.max_repeat )
                yd_hot = tf.one_hot( yd, self.max_repeat )
                ym_hot = tf.one_hot( ym, self.max_repeat )

                y1_hot = tf.one_hot( y1, self.max_repeat )
                y2_hot = tf.one_hot( y2, self.max_repeat )
                y3_hot = tf.one_hot( y3, self.max_repeat )
                y4_hot = tf.one_hot( y4, self.max_repeat )

                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # other tasks
                    s_floss, mloss_u, mloss_d, mloss_i, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m =\
                        other_tasks_loss_v2( xs1, yu_hot, yd_hot, ym_hot )

                    # other tasks
                    s_floss1, mloss_1, mloss_2, mloss_3, mloss_4,\
                    pred_1, pred_2, pred_3, pred_4 = other_tasks_loss_v1( xs2, y1_hot, y2_hot, y3_hot, y4_hot )

                    # final loss
                    floss = m_floss + s_floss + s_floss1

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                self.train_loss_tu( mloss_u )
                self.train_acc_tu.update_state( yu, tf.argmax( pred_u, axis = -1 ) )

                self.train_loss_ti( mloss_i )
                self.train_acc_ti.update_state( ym, tf.argmax( pred_m, axis = -1 ) )

                self.train_loss_td( mloss_d )
                self.train_acc_td.update_state( yd, tf.argmax( pred_d, axis = -1 ) )

                self.train_loss_t1( mloss_1 )
                self.train_acc_t1.update_state( y1, tf.argmax( pred_1, axis = -1 ) )

                self.train_loss_t2( mloss_2 )
                self.train_acc_t2.update_state( y2, tf.argmax( pred_2, axis = -1 ) )

                self.train_loss_t3( mloss_3 )
                self.train_acc_t3.update_state( y3, tf.argmax( pred_3, axis = -1 ) )

                self.train_loss_t4( mloss_4 )
                self.train_acc_t4.update_state( y4, tf.argmax( pred_4, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_total.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_t )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/u', self.train_loss_tu.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/u', self.train_acc_tu.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/i', self.train_loss_ti.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/i', self.train_acc_ti.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/d', self.train_loss_td.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/d', self.train_acc_td.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/1', self.train_loss_t1.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/1', self.train_acc_t1.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/2', self.train_loss_t2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/2', self.train_acc_t2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/3', self.train_loss_t3.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/3', self.train_acc_t3.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/4', self.train_loss_t4.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/4', self.train_acc_t4.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary/loss_u_r', rloss_u, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_i_r', rloss_m, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_d_r', rloss_d, step = self.t_step_t )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_t += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() + s_floss.numpy() + s_floss1.numpy() )

class CardsEmbeding_v3( Module ):

    def __init__(self, name, log_dir, num_cards, embeding_size, hand_size, max_repeat, num_blocks, num_heads):
    
        super(CardsEmbeding_v3, self).__init__()

        self.module_type = 'CardsEmbeding_v2'
        self.m_name = name
        max_repeat = 2

        self.log_dir = log_dir + '/' + name

        self.num_cards = num_cards
        self.embeding_size = embeding_size
        self.max_repeat = max_repeat

        # model
        # embeding base
        self.embeding = tf.Variable( tf.random.normal( [ num_cards, embeding_size ], 0, 0.02 ), trainable = True, name = 'base_cards_embeding' )
        self.layer = TransformerLayer( embeding_size, num_heads, num_blocks, hand_size, 'base_tlayer' )
        self.l1 = Dense( embeding_size, name = 'base_anchor' )
        
        # base embeding tasks
        self.isin = Dense( embeding_size, name = 'base_h_isin', activation = tanh )

        # embeding tasks
        self.sub = Dense( embeding_size, name = 'h_sub', activation = tanh )
        self.add = Dense( embeding_size, name = 'h_add', activation = tanh )
        self.mul = Dense( embeding_size, name = 'h_mul', activation = tanh )
        
        # output sets
        self.out_isin = Dense( num_cards * max_repeat, name = 'base_o_isin' )

        # train variables        
        step = tf.Variable( 0, trainable = False )
        boundaries = [ 100000, 150000 ]
        values = [ 1e-0, 1e-1, 1e-2 ]
        schedule = PiecewiseConstantDecay( boundaries, values )
        lr = 2e-3 * schedule( step )
        wd = lambda: 1e-4 * schedule( step )

        self.opt = AdamW( wd, lr )

        self.train_loss2 = tf.keras.metrics.Mean( 'train_loss2', dtype = tf.float32 )
        self.train_acc2 = tf.keras.metrics.Accuracy( 'train_acc2', dtype = tf.float32 )

        self.train_loss_tu = tf.keras.metrics.Mean( 'train_loss_u', dtype = tf.float32 )
        self.train_acc_tu = tf.keras.metrics.Accuracy( 'train_acc_u', dtype = tf.float32 )

        self.train_loss_ti = tf.keras.metrics.Mean( 'train_loss_i', dtype = tf.float32 )
        self.train_acc_ti = tf.keras.metrics.Accuracy( 'train_acc_i', dtype = tf.float32 )

        self.train_loss_td = tf.keras.metrics.Mean( 'train_loss_d', dtype = tf.float32 )
        self.train_acc_td = tf.keras.metrics.Accuracy( 'train_acc_d', dtype = tf.float32 )

        self.train_loss_t1 = tf.keras.metrics.Mean( 'train_loss_1', dtype = tf.float32 )
        self.train_acc_t1 = tf.keras.metrics.Accuracy( 'train_acc_1', dtype = tf.float32 )

        self.train_loss_t2 = tf.keras.metrics.Mean( 'train_loss_2', dtype = tf.float32 )
        self.train_acc_t2 = tf.keras.metrics.Accuracy( 'train_acc_2', dtype = tf.float32 )

        self.train_loss_t3 = tf.keras.metrics.Mean( 'train_loss_3', dtype = tf.float32 )
        self.train_acc_t3 = tf.keras.metrics.Accuracy( 'train_acc_3', dtype = tf.float32 )

        self.train_loss_t4 = tf.keras.metrics.Mean( 'train_loss_4', dtype = tf.float32 )
        self.train_acc_t4 = tf.keras.metrics.Accuracy( 'train_acc_4', dtype = tf.float32 )

        self.train_summary_writer_base = tf.summary.create_file_writer( self.log_dir + '/base' )
        self.train_summary_writer_total = tf.summary.create_file_writer( self.log_dir + '/total' )

        self.t_step = 0
        self.t_step_b = 0
        self.t_step_t = 0
        
        in1 = tf.zeros( [ 1, hand_size ], dtype = tf.int32 )
        self.__load__( in1 )

    def __load__(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        
        # anchor
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, _ = self.layer( emb, eval )

        # count cards
        h_isin = self.isin( h )
        self.out_isin( h_isin )

        # sets operations
        # h_sets = self.sets( h )
        h_sets = h_isin

        self.sub( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.add( tf.concat( [ h_sets, h_sets ], -1 ) )
        self.mul( tf.concat( [ h_sets, h_sets ], -1 ) )

    def isin_op(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, msk = self.layer( emb, eval )

        # isin cards
        h_isin = self.isin( h )
        h1 = self.out_isin( h_isin )
        h1 = tf.reshape( h1, [ -1, input.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h1 )

        return out_isin, ac, msk

    def get_embeding(self, input, eval=False):
        emb = tf.nn.embedding_lookup( self.embeding, input )
        h, _ = self.layer( emb, eval )
        return self.isin( h ) #, self.sets( h )

    def add_op(self, a, b):
        return self.add( tf.concat( [ a, b ], -1 ) )

    def sub_op(self, a, b):
        return self.sub( tf.concat( [ a, b ], -1 ) )

    def mul_op(self, a, b):
        return self.mul( tf.concat( [ a, b ], -1 ) )

    def reconstruct_sets(self, embeding, eval=False):
        h = self.out_isin( embeding )
        h = tf.reshape( h, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h )
        return out_isin

    def loss(self, x, y, w):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )# * w

        return loss

    def loss2(self, x, y):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )

        return loss

    def train_base(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, ac, msk = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        vars = [ x for x in self.trainable_variables if 'base' in x.name ]
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):
       
                x, y, idx, w = ds.sample_batch( bs )

                y_hot = tf.one_hot( y, self.max_repeat )
                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # final loss
                    floss = m_floss

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_base.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_b )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step_b )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step_b )

                    tf.summary.histogram( 'total/priorities', ds.priorities, step = self.t_step_b )

                    for i, b in enumerate(msk):
                        m = b[0]
                        for j, h in enumerate(m):
                            tf.summary.image( 'train_mosaic/b{}h{}'.format(i,j), 
                                                h[tf.newaxis,:,:,tf.newaxis], step = self.t_step_b, max_outputs = 1 )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_b += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() )

    def train(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, ac, msk = self.isin_op( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        def other_tasks_loss_v1(xs, y1_hot, y2_hot, y3_hot, y4_hot):
            
            xa, xb, xc = tf.split( xs, 3, axis = 0 )
            xa, xb, xc = tf.squeeze( xa, 0 ), tf.squeeze( xb, 0 ), tf.squeeze( xc, 0 )

            # get embedings
            out_a = self.get_embeding( xa )
            out_b = self.get_embeding( xb )
            out_c = self.get_embeding( xc )

            # set operations
            t1  = self.add_op( self.sub_op( out_a, out_b ), out_c )
            t1a = self.add_op( out_c, self.sub_op( out_a, out_b ) )
            t2  = self.sub_op( self.add_op( out_a, out_b ), out_c )
            t3  = self.sub_op( out_c, self.add_op( out_a, out_b ) )
            t4  = self.sub_op( out_c, self.sub_op( out_a, out_b ) )

            # reconstruct, but without gradients
            pred_1  = self.reconstruct_sets( t1 )
            pred_1a = self.reconstruct_sets( t1a )
            pred_2  = self.reconstruct_sets( t2 )
            pred_3  = self.reconstruct_sets( t3 )
            pred_4  = self.reconstruct_sets( t4 )
            
            # compute operation looses
            mloss_1  = tf.reduce_mean( self.loss2( pred_1,  y1_hot ) )
            mloss_1a = tf.reduce_mean( self.loss2( pred_1a, y1_hot ) )
            mloss_2  = tf.reduce_mean( self.loss2( pred_2,  y2_hot ) )
            mloss_3  = tf.reduce_mean( self.loss2( pred_3,  y3_hot ) )
            mloss_4  = tf.reduce_mean( self.loss2( pred_4,  y4_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( ( out_a + out_b + out_c ) / 3.0 )

            pr1  = tf.reduce_mean( t1  )
            pr1a = tf.reduce_mean( t1a )
            pr2  = tf.reduce_mean( t2  )
            pr3  = tf.reduce_mean( t3  )
            pr4  = tf.reduce_mean( t4  )

            rloss_1  = ( yr - pr1  ) ** 2
            rloss_1a = ( yr - pr1a ) ** 2
            rloss_2  = ( yr - pr2  ) ** 2
            rloss_3  = ( yr - pr3  ) ** 2
            rloss_4  = ( yr - pr4  ) ** 2

            # sum all looses
            s_floss = ( mloss_1 + mloss_1a + mloss_2 + mloss_3 + mloss_4 ) + 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )
            # s_floss = 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )

            return s_floss, mloss_1 + mloss_1a, mloss_2, mloss_3, mloss_4,\
                   pred_1, pred_2, pred_3, pred_4

        def other_tasks_loss_v2( xs, yu_hot, yd_hot, ym_hot):

            # get embedings
            _, sz, _ = shape_list( xs )
            xs_r = [ tf.squeeze( x, axis = 1 ) for x in tf.split( xs, sz, axis = 1 ) ]
            out_a = [ self.get_embeding( x ) for x in xs_r ]

            # union
            t_u = out_a[0]
            for x in out_a[1:]: t_u = self.add_op( t_u, x )

            # diference
            t_d = out_a[0]
            for x in out_a[1:]: t_d = self.sub_op( t_d, x )

            # intersection
            t_m = out_a[0]
            for x in out_a[1:]: t_m = self.mul_op( t_m, x )

            # reconstruct, but without gradients
            pred_u = self.reconstruct_sets( t_u )
            pred_d = self.reconstruct_sets( t_d )
            pred_m = self.reconstruct_sets( t_m )
            
            # compute operation looses
            mloss_u = tf.reduce_mean( self.loss2( pred_u, yu_hot ) )
            mloss_d = tf.reduce_mean( self.loss2( pred_d, yd_hot ) )
            mloss_m = tf.reduce_mean( self.loss2( pred_m, ym_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( out_a )

            pru = tf.reduce_mean( t_u )
            prd = tf.reduce_mean( t_d )
            prm = tf.reduce_mean( t_m )

            rloss_u = ( yr - pru ) ** 2
            rloss_d = ( yr - prd ) ** 2
            rloss_m = ( yr - prm ) ** 2

            # sum all looses
            s_floss = ( mloss_u + mloss_d + mloss_m ) + 0.2 * ( rloss_u + rloss_d + rloss_m )
            # s_floss = mloss_d + 0.2 * ( rloss_u + rloss_d + rloss_m )

            return s_floss, mloss_u, mloss_d, mloss_m, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m

        vars = self.trainable_variables
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):

                sz = randint( 2, 10 )                
                x, y, xs1, yu, yd, ym,\
                xs2, y1, y2, y3, y4,\
                idx, w = ds.sample_batch_v2( bs, sz )

                y_hot = tf.one_hot( y, self.max_repeat )
                yu_hot = tf.one_hot( yu, self.max_repeat )
                yd_hot = tf.one_hot( yd, self.max_repeat )
                ym_hot = tf.one_hot( ym, self.max_repeat )

                y1_hot = tf.one_hot( y1, self.max_repeat )
                y2_hot = tf.one_hot( y2, self.max_repeat )
                y3_hot = tf.one_hot( y3, self.max_repeat )
                y4_hot = tf.one_hot( y4, self.max_repeat )

                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # other tasks
                    s_floss, mloss_u, mloss_d, mloss_i, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m =\
                        other_tasks_loss_v2( xs1, yu_hot, yd_hot, ym_hot )

                    # other tasks
                    s_floss1, mloss_1, mloss_2, mloss_3, mloss_4,\
                    pred_1, pred_2, pred_3, pred_4 = other_tasks_loss_v1( xs2, y1_hot, y2_hot, y3_hot, y4_hot )

                    # final loss
                    floss = m_floss + s_floss + s_floss1

                # compute grads
                grads = tape.gradient( floss, vars )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                self.train_loss_tu( mloss_u )
                self.train_acc_tu.update_state( yu, tf.argmax( pred_u, axis = -1 ) )

                self.train_loss_ti( mloss_i )
                self.train_acc_ti.update_state( ym, tf.argmax( pred_m, axis = -1 ) )

                self.train_loss_td( mloss_d )
                self.train_acc_td.update_state( yd, tf.argmax( pred_d, axis = -1 ) )

                self.train_loss_t1( mloss_1 )
                self.train_acc_t1.update_state( y1, tf.argmax( pred_1, axis = -1 ) )

                self.train_loss_t2( mloss_2 )
                self.train_acc_t2.update_state( y2, tf.argmax( pred_2, axis = -1 ) )

                self.train_loss_t3( mloss_3 )
                self.train_acc_t3.update_state( y3, tf.argmax( pred_3, axis = -1 ) )

                self.train_loss_t4( mloss_4 )
                self.train_acc_t4.update_state( y4, tf.argmax( pred_4, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer_total.as_default():
                    
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step_t )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/u', self.train_loss_tu.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/u', self.train_acc_tu.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/i', self.train_loss_ti.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/i', self.train_acc_ti.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary_loss/d', self.train_loss_td.result(), step = self.t_step_t )
                    tf.summary.scalar( 'secondary_acc/d', self.train_acc_td.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/1', self.train_loss_t1.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/1', self.train_acc_t1.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/2', self.train_loss_t2.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/2', self.train_acc_t2.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/3', self.train_loss_t3.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/3', self.train_acc_t3.result(), step = self.t_step_t )

                    tf.summary.scalar( 'complex_loss/4', self.train_loss_t4.result(), step = self.t_step_t )
                    tf.summary.scalar( 'complex_acc/4', self.train_acc_t4.result(), step = self.t_step_t )

                    tf.summary.scalar( 'secondary/loss_u_r', rloss_u, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_i_r', rloss_m, step = self.t_step_t )
                    tf.summary.scalar( 'secondary/loss_d_r', rloss_d, step = self.t_step_t )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1
                self.t_step_t += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() + s_floss.numpy() + s_floss1.numpy() )

class CardsEmbeding_v2( Module ):

    def __init__(self, name, log_dir, num_cards, embeding_size, hand_size, max_repeat, num_blocks, num_heads):
    
        super(CardsEmbeding_v2, self).__init__()

        self.module_type = 'CardsEmbeding_v2'
        self.m_name = name
        max_repeat = 2

        self.log_dir = log_dir + '/' + name

        self.num_cards = num_cards
        self.embeding_size = embeding_size
        self.max_repeat = max_repeat

        # model
        # embeding base
        self.embeding = tf.Variable( tf.random.normal( [ num_cards, embeding_size ], 0, 0.02 ), trainable = True, name = 'cards_embeding' )
        self.layer = TransformerLayer( embeding_size, num_heads, num_blocks, hand_size, 'tlayer' )
        self.l1 = Dense( embeding_size, name = 'anchor' )
        
        # embeding tasks
        self.isin = Dense( embeding_size, name = 'isin', activation = tanh )
        # self.sets = Dense( embeding_size, name = 'sets', activation = None )
        self.sets = Nalu( embeding_size, name = 'sets' )
        
        # output sets
        self.out_isin = Dense( num_cards * max_repeat, name = 'isin' )
        self.out_sets = Dense( num_cards * max_repeat, name = 'sets' )

        # train variables        
        step = tf.Variable( 0, trainable = False )
        boundaries = [ 10000, 15000 ]
        values = [ 1e-0, 1e-1, 1e-2 ]
        schedule = PiecewiseConstantDecay( boundaries, values )
        lr = 2e-3 * schedule( step )
        wd = lambda: 1e-4 * schedule( step )

        self.opt = AdamW( wd, lr )

        self.train_loss2 = tf.keras.metrics.Mean( 'train_loss2', dtype = tf.float32 )
        self.train_acc2 = tf.keras.metrics.Accuracy( 'train_acc2', dtype = tf.float32 )

        self.train_loss_tu = tf.keras.metrics.Mean( 'train_loss_u', dtype = tf.float32 )
        self.train_acc_tu = tf.keras.metrics.Accuracy( 'train_acc_u', dtype = tf.float32 )

        self.train_loss_ti = tf.keras.metrics.Mean( 'train_loss_i', dtype = tf.float32 )
        self.train_acc_ti = tf.keras.metrics.Accuracy( 'train_acc_i', dtype = tf.float32 )

        self.train_loss_td = tf.keras.metrics.Mean( 'train_loss_d', dtype = tf.float32 )
        self.train_acc_td = tf.keras.metrics.Accuracy( 'train_acc_d', dtype = tf.float32 )

        self.train_loss_t1 = tf.keras.metrics.Mean( 'train_loss_1', dtype = tf.float32 )
        self.train_acc_t1 = tf.keras.metrics.Accuracy( 'train_acc_1', dtype = tf.float32 )

        self.train_loss_t2 = tf.keras.metrics.Mean( 'train_loss_2', dtype = tf.float32 )
        self.train_acc_t2 = tf.keras.metrics.Accuracy( 'train_acc_2', dtype = tf.float32 )

        self.train_loss_t3 = tf.keras.metrics.Mean( 'train_loss_3', dtype = tf.float32 )
        self.train_acc_t3 = tf.keras.metrics.Accuracy( 'train_acc_3', dtype = tf.float32 )

        self.train_loss_t4 = tf.keras.metrics.Mean( 'train_loss_4', dtype = tf.float32 )
        self.train_acc_t4 = tf.keras.metrics.Accuracy( 'train_acc_4', dtype = tf.float32 )

        self.train_summary_writer = tf.summary.create_file_writer( self.log_dir )

        self.t_step = 0
        
        im = tf.zeros( [ 1, hand_size ], dtype = tf.int32 )
        self( im )

    def __call__(self, input, eval=False):
        
        # get cards embeding
        emb = tf.nn.embedding_lookup( self.embeding, input )
        ac = self.l1( emb )
        ac = softmax( ac @ tf.transpose( self.embeding ) )

        # compute hand context
        h, msk = self.layer( emb, eval )

        # count cards
        h_isin = self.isin( h )
        h1 = self.out_isin( h_isin )
        h1 = tf.reshape( h1, [ -1, input.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h1 )

        # sets operations
        h_sets = self.sets( h )
        h2 = self.out_sets( h_sets )
        h2 = tf.reshape( h2, [ -1, input.shape[1], self.num_cards, self.max_repeat ] )
        out_sets = softmax( h2 )

        return out_isin, out_sets, ac, msk

    def get_embeding(self, input, eval=False):
        emb = tf.nn.embedding_lookup( self.embeding, input )
        h, _ = self.layer( emb, eval )
        return self.isin( h ), self.sets( h )

    def reconstruct_sets(self, embeding, eval=False):
        h = self.out_sets( embeding )
        h = tf.reshape( h, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_sets = softmax( h )
        return out_sets

    def get_embeding_eval(self, input, eval=False):
        emb = tf.nn.embedding_lookup( self.embeding, input )
        h, _ = self.layer( emb, eval )
        return tanh( self.h1( h[:,-1,:] ) )

    def reconstruct_eval(self, embeding, eval=False):
        
        h1 = self.out_isin( embeding )
        h1 = tf.reshape( h1, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_isin = softmax( h1 )

        h2 = self.out_sets( embeding )
        h2 = tf.reshape( h2, [ -1, embeding.shape[1], self.num_cards, self.max_repeat ] )
        out_sets = softmax( h2 )
        
        return out_isin, tf.argmax( out_isin, -1 ), out_sets, tf.argmax( out_sets, -1 ), 

    def loss(self, x, y, w):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )# * w

        return loss

    def loss2(self, x, y):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 )

        return loss

    def train(self, ds, new_ds, bs, epochs):

        def main_loss(x, y_hot):

            out_isin, _, ac, msk = self( x )                    
            tloss = self.loss( out_isin, y_hot, w ) 
            ls = tf.reduce_mean( tloss ) + 0.5 * tf.reduce_mean( sparse_categorical_crossentropy( x, ac ) )

            return ls, tloss, out_isin, msk

        def other_tasks_loss_v1(xs, y1_hot, y2_hot, y3_hot, y4_hot):
            
            xa, xb, xc = tf.split( xs, 3, axis = 0 )
            xa, xb, xc = tf.squeeze( xa, 0 ), tf.squeeze( xb, 0 ), tf.squeeze( xc, 0 )

            # get embedings
            _, out_a = self.get_embeding( xa )
            _, out_b = self.get_embeding( xb )
            _, out_c = self.get_embeding( xc )

            # set operations
            t1  = ( out_a - out_b ) + out_c
            t1a = out_c + ( out_a - out_b )

            t2  = ( out_a + out_b ) - out_c #
            t3  = out_c - ( out_a + out_b )

            t4  = out_c - ( out_a - out_b ) #

            # reconstruct, but without gradients
            pred_1  = self.reconstruct_sets( t1 )
            pred_1a = self.reconstruct_sets( t1a )
            pred_2  = self.reconstruct_sets( t2 )
            pred_3  = self.reconstruct_sets( t3 )
            pred_4  = self.reconstruct_sets( t4 )
            
            # compute operation looses
            mloss_1  = tf.reduce_mean( self.loss2( pred_1,  y1_hot ) )
            mloss_1a = tf.reduce_mean( self.loss2( pred_1a, y1_hot ) )
            mloss_2  = tf.reduce_mean( self.loss2( pred_2,  y2_hot ) )
            mloss_3  = tf.reduce_mean( self.loss2( pred_3,  y3_hot ) )
            mloss_4  = tf.reduce_mean( self.loss2( pred_4,  y4_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( ( out_a + out_b + out_c ) / 3.0 )

            pr1  = tf.reduce_mean( t1  )
            pr1a = tf.reduce_mean( t1a )
            pr2  = tf.reduce_mean( t2  )
            pr3  = tf.reduce_mean( t3  )
            pr4  = tf.reduce_mean( t4  )

            rloss_1  = ( yr - pr1  ) ** 2
            rloss_1a = ( yr - pr1a ) ** 2
            rloss_2  = ( yr - pr2  ) ** 2
            rloss_3  = ( yr - pr3  ) ** 2
            rloss_4  = ( yr - pr4  ) ** 2

            # sum all looses
            # s_floss = ( mloss_1 + mloss_1a + mloss_2 + mloss_3 + mloss_4 ) + 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )
            s_floss = 0.2 * ( rloss_1 + rloss_1a + rloss_2 + rloss_3 + rloss_4 )

            return s_floss, mloss_1 + mloss_1a, mloss_2, mloss_3, mloss_4,\
                   pred_1, pred_2, pred_3, pred_4

        def other_tasks_loss_v2( xs, yu_hot, yd_hot, ym_hot):

            # get embedings
            _, sz, _ = shape_list( xs )
            xs_r = [ tf.squeeze( x, axis = 1 ) for x in tf.split( xs, sz, axis = 1 ) ]
            out_a = [ self.get_embeding( x )[1] for x in xs_r ]

            # union
            t_u = out_a[0]
            for x in out_a[1:]: t_u += x

            # diference
            t_d = out_a[0]
            for x in out_a[1:]: t_d -= x

            # intersection
            t_m = out_a[0]
            for x in out_a[1:]: t_m *= x

            # reconstruct, but without gradients
            pred_u = self.reconstruct_sets( t_u )
            pred_d = self.reconstruct_sets( t_d )
            pred_m = self.reconstruct_sets( t_m )
            
            # compute operation looses
            mloss_u = tf.reduce_mean( self.loss2( pred_u, yu_hot ) )
            mloss_d = tf.reduce_mean( self.loss2( pred_d, yd_hot ) )
            mloss_m = tf.reduce_mean( self.loss2( pred_m, ym_hot ) )

            # compute anchor loss to keep embedings in the same space
            yr = tf.reduce_mean( out_a )

            pru = tf.reduce_mean( t_u )
            prd = tf.reduce_mean( t_d )
            prm = tf.reduce_mean( t_m )

            rloss_u = ( yr - pru ) ** 2
            rloss_d = ( yr - prd ) ** 2
            rloss_m = ( yr - prm ) ** 2

            # sum all looses
            # s_floss = ( mloss_u + mloss_d + mloss_m ) + 0.2 * ( rloss_u + rloss_d + rloss_m )
            s_floss = mloss_d + 0.2 * ( rloss_u + rloss_d + rloss_m )

            return s_floss, mloss_u, mloss_d, mloss_m, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m

        vars = self.trainable_variables
        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):

                sz = randint( 2, 10 )                
                x, y, xs1, yu, yd, ym,\
                xs2, y1, y2, y3, y4,\
                idx, w = ds.sample_batch_v2( bs, sz )

                y_hot = tf.one_hot( y, self.max_repeat )
                yu_hot = tf.one_hot( yu, self.max_repeat )
                yd_hot = tf.one_hot( yd, self.max_repeat )
                ym_hot = tf.one_hot( ym, self.max_repeat )

                y1_hot = tf.one_hot( y1, self.max_repeat )
                y2_hot = tf.one_hot( y2, self.max_repeat )
                y3_hot = tf.one_hot( y3, self.max_repeat )
                y4_hot = tf.one_hot( y4, self.max_repeat )

                with tf.GradientTape() as tape:
                    
                    # main operation
                    m_floss, tloss, out, msk = main_loss( x, y_hot )
                    
                    # other tasks
                    s_floss, mloss_u, mloss_d, mloss_i, rloss_u, rloss_d, rloss_m, pred_u, pred_d, pred_m =\
                        other_tasks_loss_v2( xs1, yu_hot, yd_hot, ym_hot )

                    # other tasks
                    s_floss1, mloss_1, mloss_2, mloss_3, mloss_4,\
                    pred_1, pred_2, pred_3, pred_4 = other_tasks_loss_v1( xs2, y1_hot, y2_hot, y3_hot, y4_hot )

                    # final loss
                    floss = m_floss + s_floss + s_floss1

                # compute grads
                grads = tape.gradient( floss, vars )

                # get global norm 
                # norm = tf.linalg.global_norm( grads )

                # clip gradients by norm
                # grads_norm, _ = tf.clip_by_global_norm( grads, 0.2, norm )

                # apply grads
                self.opt.apply_gradients( zip( grads, vars ) )

                # store metrices
                self.train_loss2( m_floss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                self.train_loss_tu( mloss_u )
                self.train_acc_tu.update_state( yu, tf.argmax( pred_u, axis = -1 ) )

                self.train_loss_ti( mloss_i )
                self.train_acc_ti.update_state( ym, tf.argmax( pred_m, axis = -1 ) )

                self.train_loss_td( mloss_d )
                self.train_acc_td.update_state( yd, tf.argmax( pred_d, axis = -1 ) )

                self.train_loss_t1( mloss_1 )
                self.train_acc_t1.update_state( y1, tf.argmax( pred_1, axis = -1 ) )

                self.train_loss_t2( mloss_2 )
                self.train_acc_t2.update_state( y2, tf.argmax( pred_2, axis = -1 ) )

                self.train_loss_t3( mloss_3 )
                self.train_acc_t3.update_state( y3, tf.argmax( pred_3, axis = -1 ) )

                self.train_loss_t4( mloss_4 )
                self.train_acc_t4.update_state( y4, tf.argmax( pred_4, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer.as_default():
                    
                    tf.summary.scalar( 'total/sz', sz, step = self.t_step )
                    tf.summary.scalar( 'total/floss', floss, step = self.t_step )

                    tf.summary.scalar( 'total/loss', self.train_loss2.result(), step = self.t_step )
                    tf.summary.scalar( 'total/acc', self.train_acc2.result(), step = self.t_step )

                    tf.summary.scalar( 'secondary_loss/u', self.train_loss_tu.result(), step = self.t_step )
                    tf.summary.scalar( 'secondary_acc/u', self.train_acc_tu.result(), step = self.t_step )

                    tf.summary.scalar( 'secondary_loss/i', self.train_loss_ti.result(), step = self.t_step )
                    tf.summary.scalar( 'secondary_acc/i', self.train_acc_ti.result(), step = self.t_step )

                    tf.summary.scalar( 'secondary_loss/d', self.train_loss_td.result(), step = self.t_step )
                    tf.summary.scalar( 'secondary_acc/d', self.train_acc_td.result(), step = self.t_step )

                    tf.summary.scalar( 'complex_loss/1', self.train_loss_t1.result(), step = self.t_step )
                    tf.summary.scalar( 'complex_acc/1', self.train_acc_t1.result(), step = self.t_step )

                    tf.summary.scalar( 'complex_loss/2', self.train_loss_t2.result(), step = self.t_step )
                    tf.summary.scalar( 'complex_acc/2', self.train_acc_t2.result(), step = self.t_step )

                    tf.summary.scalar( 'complex_loss/3', self.train_loss_t3.result(), step = self.t_step )
                    tf.summary.scalar( 'complex_acc/3', self.train_acc_t3.result(), step = self.t_step )

                    tf.summary.scalar( 'complex_loss/4', self.train_loss_t4.result(), step = self.t_step )
                    tf.summary.scalar( 'complex_acc/4', self.train_acc_t4.result(), step = self.t_step )

                    tf.summary.scalar( 'secondary/loss_u_r', rloss_u, step = self.t_step )
                    tf.summary.scalar( 'secondary/loss_i_r', rloss_m, step = self.t_step )
                    tf.summary.scalar( 'secondary/loss_d_r', rloss_d, step = self.t_step )

                    tf.summary.histogram( 'total/priorities', ds.priorities, step = self.t_step )

                    for i, b in enumerate(msk):
                        m = b[0]
                        for j, h in enumerate(m):
                            tf.summary.image( 'train_mosaic/b{}h{}'.format(i,j), 
                                                h[tf.newaxis,:,:,tf.newaxis], step = self.t_step, max_outputs = 1 )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss.numpy() + s_floss.numpy() + s_floss1.numpy() )

class CardsEmbeding( Module ):

    def __init__(self, name, log_dir, num_cards, embeding_size, hand_size, max_repeat, num_blocks, num_heads):
    
        super(CardsEmbeding, self).__init__()

        self.module_type = 'CardsEmbeding'
        self.m_name = name
        max_repeat = 2

        self.log_dir = log_dir + '/' + name

        self.num_cards = num_cards
        self.embeding_size = embeding_size
        self.max_repeat = max_repeat

        # model
        self.embeding = tf.Variable( tf.random.normal( [ num_cards, embeding_size ], 0, 0.02 ), trainable = True, name = 'cards_embeding' )
        self.layer = TransformerLayer( embeding_size, num_heads, num_blocks, hand_size, 'tlayer' )
        self.l1 = Dense( num_cards )
        self.out = Dense( num_cards * max_repeat )

        # train variables
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
            3e-4, 100, 0.97, staircase = False
        )
        self.opt = Adam( tf.Variable( 2e-4 ) )

        self.train_loss1 = tf.keras.metrics.Mean( 'train_loss1', dtype = tf.float32 )
        self.train_loss2 = tf.keras.metrics.Mean( 'train_loss2', dtype = tf.float32 )

        self.train_l2_loss = tf.keras.metrics.Mean( 'train_l2_loss', dtype = tf.float32 )

        self.train_acc1 = tf.keras.metrics.Accuracy( 'train_acc1', dtype = tf.float32 )
        self.train_acc2 = tf.keras.metrics.Accuracy( 'train_acc2', dtype = tf.float32 )

        self.train_summary_writer = tf.summary.create_file_writer( self.log_dir )

        self.t_step = 0
        
        im = tf.zeros( [ 1, hand_size ], dtype = tf.int32 )
        self( im )

    def __call__(self, input, eval=False):
        
        # get cards embeding
        emb = tf.gather( self.embeding, input )

        # compute hand context
        h, msk = self.layer( emb, eval )

        # count cards
        out_emb = tf.matmul( h, self.embeding, transpose_b = True )
        h0 = self.l1( h )
        h1 = self.out( h0 + out_emb )
        h1 = tf.reshape( h1, [ -1, input.shape[1], self.num_cards, self.max_repeat ] )
        out = softmax( h1 )

        return out, msk

    def get_embeding(self, input, eval=False):

        # get cards embeding
        emb = tf.gather( self.embeding, input )

        # compute hand context
        h, _ = self.layer( emb, eval )

        return h[:,-1,:]

    def reconstruct(self, embeding, eval=False):

        out_emb = tf.matmul( embeding, self.embeding, transpose_b = True )

        # count cards
        h0 = self.l1( embeding )

        h1 = self.out( h0 + out_emb )
        h1 = tf.reshape( h1, [ -1, self.num_cards, self.max_repeat ] )
        out = softmax( h1 )

        return out, tf.argmax( out, -1 )

    def loss(self, x, y, w):

        loss = binary_crossentropy( y, x, from_logits = False )
        loss = tf.reduce_mean( tf.reduce_mean( loss, -1 ), -1 ) * w

        return loss

    def train(self, ds, new_ds, bs, epochs):

        vars = self.trainable_variables
        l2_vars = [ vr for vr in vars if not 'norm' in vr.name ]

        for _ in range(epochs):

            dsx, dsy = new_ds( 1000 )
            [ ds.store( x, y ) for x, y in zip( dsx, dsy ) ]

            for d in tqdm( range( 200 ) ):

                x, y, idx, w = ds.sample_batch( bs )
                y_hot = tf.one_hot( y, self.max_repeat )
                with tf.GradientTape() as tape:
                    
                    out, msk = self( x )
                    
                    tloss2 = self.loss( out, y_hot, w )                    
                    mloss2 = tf.reduce_mean( tloss2 )
                    
                    l2_loss = 2e-6 * tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in l2_vars ] )

                    floss = mloss2 + l2_loss

                grads = tape.gradient( floss, vars )

                # get global norm 
                norm = tf.linalg.global_norm( grads )

                # clip gradients by norm
                grads_norm, _ = tf.clip_by_global_norm( grads, 0.2, norm )

                # apply grads
                self.opt.apply_gradients( zip( grads_norm, vars ) )

                # store metrices
                self.train_loss2( mloss2 )
                self.train_l2_loss( l2_loss )
                self.train_acc2.update_state( y, tf.argmax( out, axis = -1 ) )

                # log on tensorboard
                with self.train_summary_writer.as_default():
                    
                    # tf.summary.scalar( 'total/loss1', self.train_loss1.result(), step = self.t_step )
                    tf.summary.scalar( 'total/loss2', self.train_loss2.result(), step = self.t_step )

                    # tf.summary.scalar( 'total/acc1', self.train_acc1.result(), step = self.t_step )
                    tf.summary.scalar( 'total/acc2', self.train_acc2.result(), step = self.t_step )

                    tf.summary.scalar( 'total/l2', self.train_l2_loss.result(), step = self.t_step )

                    tf.summary.histogram( 'total/priorities', ds.priorities, step = self.t_step )

                    for i, b in enumerate(msk):
                        m = b[0]
                        for j, h in enumerate(m):
                            tf.summary.image( 'train_mosaic/b{}h{}'.format(i,j), 
                                                h[tf.newaxis,:,:,tf.newaxis], step = self.t_step, max_outputs = 1 )
                
                # save model
                if self.t_step%100 == 0:
                    self.save_tarining( 'saved/', self.m_name )

                self.t_step += 1

                # update dataset priorities
                ds.update_priorities( idx, tloss2 )