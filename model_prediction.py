from model.aggregate import aggregate_wbg_channel as aggregate
import torch
import numpy as np
import os

class CustomModelPrediction(object):
  def __init__(self, model):
    self._model = model

  def predict(self, instances, **kwargs):
    _, mask = aggregate(torch.sigmoid(self._model(instances)))
    np_mask = (mask.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8)

    return np_mask

  @classmethod
  def from_path(cls, model_dir):
    import torch
    from model.network import deeplabv3plus_resnet50 as S2M

    model = S2M()
    model_path = os.path.join(model_dir, 's2m.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.eval()
    torch.set_grad_enabled(False)
    return cls(model)
