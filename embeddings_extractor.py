import torch
import numpy as np


def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  repo = 'OxWearables/ssl-wearables'
  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  return feature_encoder


def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    acc_xyz: np.ndarray shape (N, 3) em m/s^2 (ou g), amostrado a fs_in_hz (float)
    devolve:
      acc_resampled: np.ndarray shape (M, 3) já a 30 Hz
      fs_target: 30.0
    """
    fs_target = 30.0
    win_size = 5 # in seconds
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled, fs_target


def acc_segmentation(data):
    MIN_SEGMENT_SIZE = 20
    win_size = 5000            # ms
    step = win_size // 2       # 2500
    TIMESTAMP_COL = "timestamp"
    ACTIVITY_COL = "activity_label"
    
    segments = []
    activities = []
    starts = []
    ends = []
    
    participant_end = int(np.max(data['timestamp']))
    
    for start in range(0, participant_end, step):
        end = start + win_size
        
        mask = (data[TIMESTAMP_COL] >= start) & (data[TIMESTAMP_COL] < end)
        seg = data.loc[mask]
        
        if len(seg) < MIN_SEGMENT_SIZE:
            continue
        
        unique_acts = np.unique(seg[ACTIVITY_COL])
        if len(unique_acts) != 1:
            continue
        
        if unique_acts[0] >= 8:
            continue
        
        acc_xyz = seg.loc[:, ["acce_x", "acce_y", "acce_z"]].to_numpy()
        activity = unique_acts[0]
        
        segments.append(acc_xyz)
        activities.append(activity)
        starts.append(start)
        ends.append(end)
    
    return segments, activities, starts, ends
