"""Step 6 -- load Metrica Sample Game 3 (public, open data) into lab/cache/g3.npz.

    git clone https://github.com/metrica-sports/sample-data lab/sample-data
    pip install kloppy
    python lab/prepare_tracking.py
"""
import os
import numpy as np
from kloppy import metrica

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.environ.get("METRICA_SAMPLE_DIR", os.path.join(HERE, "sample-data"))
D = os.path.join(SAMPLE, "data", "Sample_Game_3")

ds = metrica.load_tracking_epts(meta_data=os.path.join(D, "Sample_Game_3_metadata.xml"),
                                raw_data=os.path.join(D, "Sample_Game_3_tracking.txt"),
                                coordinates="metrica")
df = ds.to_df()
home = [p.player_id for p in ds.metadata.teams[0].players]
away = [p.player_id for p in ds.metadata.teams[1].players]
H = np.stack([df[[f"{p}_x", f"{p}_y"]].to_numpy(float) for p in home], 1)
A = np.stack([df[[f"{p}_x", f"{p}_y"]].to_numpy(float) for p in away], 1)
os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
np.savez_compressed(os.path.join(HERE, "cache", "g3.npz"), frame=df.frame_id.to_numpy(),
                    period=df.period_id.to_numpy(), ball=df[["ball_x", "ball_y"]].to_numpy(float),
                    H=H, A=A)
print("saved lab/cache/g3.npz", df.shape)
