from dataclasses import dataclass

"""This dataclass contains the information the RL agent recieves from the barcode classifier."""


@dataclass
class SingleBarcodeClassifierConfidence:
    confidence_score: float
    id: int


@dataclass
class AllBarcodeClassifierConfidences:
    id_to_confidence_map: dict[int, SingleBarcodeClassifierConfidence]
