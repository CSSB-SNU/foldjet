"""Protenix-v1-style finetuning of a FoldJet model.

The MiniWorld recipe on the FoldJet graph.

Phase 0/1 (this package): the recipe (:mod:`.config`), the training-mode switch
(:mod:`.prepare`), one eager micro step (:mod:`.step`: trunk with grad on the last
pass only, EDM noise + augmentation, diffusion head, diffusion / bond / distogram
loss), Adam + EMA + clip (:mod:`.optim`), native checkpoint save / resume
(:mod:`.checkpoint`) and the batch protocol (:mod:`.data`). FoldJet does not import
MiniWorld: a dataloader is plugged in as ``--batch-source module:callable`` (see
:mod:`.data`).
"""

from foldjet.train.config import LossWeights, TrainConfig
from foldjet.train.data import TrainBatch

__all__ = ["LossWeights", "TrainBatch", "TrainConfig"]
