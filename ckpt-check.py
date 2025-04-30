from EasyLM.checkpoint import StreamingCheckpointer
import flax
from flax.traverse_util import flatten_dict
import numpy as np

path = 'trainstate_params::lvm-ckpts/checkpoints-no-stream'
_, flax_params = StreamingCheckpointer.load_trainstate_checkpoint(path)
print(flax_params)