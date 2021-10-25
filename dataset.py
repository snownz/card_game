import numpy as np
import tensorflow as tf

import json
import pickle

from random import randint

def load_json(folder):
    with open(folder, "r") as f:
        return json.load( f )

def create_dataset(warper, ds_size=1024, hand_size=16, max_repeat=2):

    dsx = np.zeros( [ ds_size, hand_size ] )
    dsy_sets = np.zeros( [ ds_size, hand_size, len( warper ) ] )
    dsy_math = np.zeros( [ ds_size, hand_size, len( warper ) ] )
    for x in range( ds_size ):
        
        selected = np.zeros( len( warper ) )
        count = np.zeros( len( warper ) )
        array = ( np.random.uniform( 0, 1, hand_size ) * ( len( warper ) ) ).astype(np.int32)
        for y, card in enumerate(array):

            dsx[x,y] = card
            selected[card] = 1
            count[card] += 1
            dsy_sets[x,y] = np.copy( selected )
            dsy_math[x,y] = np.copy( count )
    
    return dsx, dsy_sets, dsy_math

def sets_tasks(data, hand_size, wp, op_size=4):

    div = len(data) // op_size
    
    xs = []
    yu = np.zeros( [ div, hand_size, wp ] )
    yd = np.zeros( [ div, hand_size, wp ] )
    ym = np.zeros( [ div, hand_size, wp ] )

    for d in range( div ):

        ids = np.arange( d * op_size, d * op_size + op_size )
        values = data[ids].astype(np.int32)

        s_u = np.zeros( wp )
        s_d = np.zeros( wp )
        s_m = np.zeros( wp )

        xs.append( values )

        for j in range( hand_size ):

            # sum
            for vl in values:
                v = int(vl[j])
                s_u[v] = 1
            yu[d,j] = np.copy( s_u )

            # dif
            s_d[values[0,j]] = 1
            reference = values[0:1,:j+1].flatten().tolist()
            total = values[1:,:j+1].flatten().tolist()
            for r in reference:
                if r in total:
                    s_d[r] = 0
            yd[d,j] = np.copy( s_d )

            # mult
            reference = values[0:1,:j+1].flatten().tolist()
            total = values[1:,:j+1].tolist()
            for r in reference:
                s_m[r] = 1
                for t in total:
                    if not r in t:
                        s_m[r] = 0
            ym[d,j] = np.copy( s_m )

    return xs, yu, yd, ym

def complex_sets_tasks(data, hand_size, wp):

    while True:
        if len(data)%3 == 1: 
            data = data[:-1]
        else:
            break
    
    div = len(data) // 3
    xa = []
    xb = []
    xc = []

    y1 = np.zeros( [ div, hand_size, wp ] )
    y2 = np.zeros( [ div, hand_size, wp ] )
    y3 = np.zeros( [ div, hand_size, wp ] )
    y4 = np.zeros( [ div, hand_size, wp ] )

    for d in range(div):
        
        init = d*3
        end = init+3
        values = data[init:end].astype(np.int32)
        
        xa.append( values[0] )
        xb.append( values[1] )
        xc.append( values[2] )

        # 1 - ( a - b ) + c or  c + ( a - b )
        s_d = np.zeros( wp ).astype(bool)
        s_u = np.zeros( wp ).astype(bool)

        for j in range( hand_size ):

            c_a = int( values[0,j] )
            c_b = int( values[1,j] )
            c_c = int( values[2,j] )

            # sub
            a = values[0,:j+1].flatten().tolist()
            b = values[1,:j+1].flatten().tolist()
            c = values[2,:j+1].flatten().tolist()

            s_d[c_a] = True
            for r in a:
                if r in b:
                    s_d[r] = False
            
            # sum
            s_u[ c_c ] = 1

            # target
            y1[d,j] = np.copy( ( s_d | s_u ) )
        
        # 2 - ( a + b ) - c
        s_d = np.zeros( wp ).astype(bool)
        s_u = np.zeros( wp ).astype(bool)

        for j in range( hand_size ):

            c_a = int( values[0,j] )
            c_b = int( values[1,j] )
            c_c = int( values[2,j] )

            # sum
            s_u[ c_a ] = 1
            s_u[ c_b ] = 1

            # sub
            c = values[2,:j+1].flatten().tolist()
            ab = values[0,:j+1].flatten().tolist() + values[1,:j+1].flatten().tolist()

            s_d[c_a] = True
            s_d[c_b] = True
            for r in ab:
                if r in c:
                    s_d[r] = False
            
            # target
            y2[d,j] = np.copy( ( s_d & s_u ) )

        # 3 - c - ( a + b )
        s_d = np.zeros( wp ).astype(bool)

        for j in range( hand_size ):

            c_a = int( values[0,j] )
            c_b = int( values[1,j] )
            c_c = int( values[2,j] )

            # sum
            s_u[ c_a ] = 1

            # sub
            c = values[2,:j+1].flatten().tolist()
            ab = values[0,:j+1].flatten().tolist() + values[1,:j+1].flatten().tolist()

            s_d[c_c] = True
            for r in c:
                if r in ab:
                    s_d[r] = False
            
            # target
            y3[d,j] = np.copy( s_d )
        
        # 4 - c - ( a - b )
        s_d1 = np.zeros( wp ).astype(bool)
        s_d2 = np.zeros( wp ).astype(bool)

        for j in range( hand_size ):

            c_a = int( values[0,j] )
            c_b = int( values[1,j] )
            c_c = int( values[2,j] )

            a = values[0,:j+1].flatten().tolist()
            b = values[1,:j+1].flatten().tolist()
            c = values[2,:j+1].flatten().tolist()

            # sub ( a - b )
            s_d1[c_a] = True
            for r in c_a:
                if r in b:
                    s_d1[r] = False
            
            # c - ( a - b )
            ab = np.where( s_d1 )[0].flatten().tolist()
            s_d2[c_c] = True
            for r in c:
                if r in ab:
                    s_d2[r] = False

            # target
            y4[d,j] = np.copy( s_d2 )

    return ( xa, xb, xc ), y1, y2, y3, y4

def math_tasks(data, hand_size, wp, op_size=4):

    div = len(data) // op_size
    
    xs = []
    yu = np.zeros( [ div, hand_size, wp ] )
    yd = np.zeros( [ div, hand_size, wp ] )

    for d in range( div ):

        ids = np.arange( d * op_size, d * op_size + op_size )
        values = data[ids].astype(np.int32)

        s_u = np.zeros( wp )
        s_d = np.zeros( wp )
        s_m = np.zeros( wp )

        xs.append( values )

        for j in range( hand_size ):

            # sum
            for vl in values:
                v = int(vl[j])
                s_u[v] += 1
            yu[d,j] = np.copy( s_u )

            # dif
            s_d[values[0,j]] += 1
            reference = values[0:1,:j+1].flatten().tolist()
            total = values[1:,:j+1].flatten().tolist()
            for r in reference:
                if r in total:
                    s_d[r] -= 1
            yd[d,j] = np.copy( s_d )

    return xs, yu, yd

class CardsWarper:

    def __init__(self, path):
        
        self.int_to_card = load_json( path )
        self.card_to_int = { self.int_to_card[x] : int(x) for x in self.int_to_card }

    def __len__(self):
        return len(self.int_to_card)

class PrioritizedReplay(object):

    def __init__(self, capacity, seed, wp, hand_size, gamma=0.99, tau=1, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        
        self.wp = wp
        self.hand_size = hand_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = [ None ] * capacity
        self.pos        = 0
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.gamma = gamma
        self.tau = tau

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, *values):
        
        # n_step calc
        max_prio = self.priorities.max() # gives max priority if buffer is not empty else 1

        self.buffer[self.pos] = values
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0

    def sample_batch(self, bs):
        
        N = len(self.buffer)
        if N == self.capacity: prios = self.priorities
        else: prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice( N, bs, p = P ) 
        samples = [ self.buffer[idx] for idx in indices ]
        
        beta = self.beta_by_frame( self.frame )
        self.frame += 1
                
        #Compute importance-sampling weight
        weights = ( N * P[indices] ) ** ( -beta )
        
        # normalize weights
        weights /= weights.max() 
        weights = np.array( weights, dtype = np.float32 ) 
        
        x = np.array( [ v[0] for v in samples ] )
        y_sets = np.array( [ v[1] for v in samples ] )
        y_math = np.array( [ v[2] for v in samples ] )

        return tf.cast( tf.convert_to_tensor( x ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y_sets ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y_math ), tf.float32 ),\
               indices, tf.convert_to_tensor( weights )

    def sample_batch_v2(self, bs, sz):
        
        N = len(self.buffer)
        if N == self.capacity: prios = self.priorities
        else: prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice( N, bs, p = P ) 
        samples = [ self.buffer[idx] for idx in indices ]
        
        beta = self.beta_by_frame( self.frame )
        self.frame += 1
                
        #Compute importance-sampling weight
        weights = ( N * P[indices] ) ** ( -beta )
        
        # normalize weights
        weights /= weights.max() 
        weights = np.array( weights, dtype = np.float32 ) 
        
        x = np.array( [ v[0] for v in samples ] )
        y = np.array( [ v[1] for v in samples ] )

        xs1, y1, y2, y3, y4 = complex_sets_tasks( x, self.hand_size, len( self.wp ) )
        xs2, yu, yd, ym = sets_tasks( x, self.hand_size, len( self.wp ), sz )

        return tf.cast( tf.convert_to_tensor( x ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y ), tf.int32 ),\
               \
               tf.cast( tf.convert_to_tensor( xs2 ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( yu ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( yd ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( ym ), tf.int32 ),\
               \
               tf.cast( tf.convert_to_tensor( xs1 ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y1 ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y2 ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y3 ), tf.int32 ),\
               tf.cast( tf.convert_to_tensor( y4 ), tf.int32 ),\
               \
               indices, tf.convert_to_tensor( weights )
      
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def save(self, path):

        with open( path + "/priorities.pkl", "wb") as f:
            pickle.dump( self.priorities, f )
        
        with open( path + "/pos.pkl", "wb") as f:
            pickle.dump( self.pos, f )

        with open( path + "/buffer.pkl", "wb") as f:
            pickle.dump( self.buffer, f )

    def load(self, path):

        with open( path + "/priorities.pkl", "rb") as f:
            self.priorities = pickle.load( f )
        
        with open( path + "/pos.pkl", "rb") as f:
            self.pos = pickle.load( f )

        with open( path + "/buffer.pkl", "rb") as f:
            self.buffer = pickle.load( f )

    def __len__(self):
        return len(self.buffer)