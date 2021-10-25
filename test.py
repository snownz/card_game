from dataset import sets_tasks_v2
import numpy as np

data = np.array( [ [ 1, 2, 3, 4, 5 ], [ 3, 4, 5, 6, 7 ] ] )
_, _, _, ym = sets_tasks_v2(data, 5, 53, 2 )