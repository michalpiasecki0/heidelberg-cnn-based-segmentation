from pathlib import Path
from dataclasses import dataclass

ROOT_PATH = Path(__file__).absolute().parent.parent
DATASETS_PATH = ROOT_PATH / "datasets"
DIFFERENTIAL_INF_PATH = DATASETS_PATH / "DIC-C2DH-HeLa"
FLUO_PATH = DATASETS_PATH / "Fluo-N2DL-HeLa"
PHASE_CONTRAST_PATH = DATASETS_PATH / "PhC-C2DH-U373"


@dataclass
class LossWeights:
    # class for weights for both losses in training
    cross_entropy: float
    dice: float


# i calculate these values in notebooks/extract_mean_std_from_datasets.ipynb
FLUO_MEAN, FLUO_STD = (33203.0, 305.0)
PHASE_MEAN, PHASE_STD = (0.32636, 0.03484)
DIC_MEAN, DIC_STD = (0.46680, 0.04071)

