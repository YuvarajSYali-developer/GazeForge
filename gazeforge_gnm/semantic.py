"""NumPy re-implementation of GNM's semantic samplers (no TensorFlow needed).

Loads the Keras decoder weights shipped with google/GNM (Apache-2.0) from the
.h5 files and evaluates the same MLP: concat([z, one_hot]) -> 4 ReLU dense
layers -> linear output.
"""
import h5py
import numpy as np

EXPRESSIONS = ["surprise", "disgust", "suck", "compress_face", "stretch_face",
               "happy", "squint", "platysma", "blow", "funneler", "smile_wide",
               "corners_down", "pucker", "wink_left", "wink_right", "mouth_left",
               "mouth_right", "lips_roll_in", "snarl", "tongue_center"]
GENDERS = ["female", "male"]
ETHNICITIES = ["middle_eastern", "asian", "white", "black"]


class Decoder:
    def __init__(self, path):
        f = h5py.File(path, "r")
        g = f["model_weights"]
        names = sorted([k for k in g.keys() if k.startswith("dense")],
                       key=lambda s: int(s.split("_")[1]))
        self.layers = [(np.array(g[n][n]["kernel:0"]), np.array(g[n][n]["bias:0"]))
                       for n in names]
        self.latent_dim = 64

    def __call__(self, z, label):
        x = np.concatenate([z, label], axis=-1).astype(np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x


def one_hot(i, n):
    v = np.zeros(n)
    v[i] = 1.0
    return v


class Semantic:
    def __init__(self, data_dir):
        self.expr = Decoder(f"{data_dir}/expression_decoder_model.h5")
        self.iden = Decoder(f"{data_dir}/identity_decoder_model.h5")

    def expression(self, name, z=None):
        z = np.zeros(64) if z is None else z
        return self.expr(z[None], one_hot(EXPRESSIONS.index(name), 20)[None])[0]

    def identity(self, gender, ethnicity, z=None):
        z = np.zeros(64) if z is None else z
        lab = np.concatenate([one_hot(GENDERS.index(gender), 2),
                              one_hot(ETHNICITIES.index(ethnicity), 4)])
        return self.iden(z[None], lab[None])[0]
