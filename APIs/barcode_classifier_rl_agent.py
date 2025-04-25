from dataclasses import dataclass

"""This dataclass contains the information the RL agent recieves from the barcode classifier."""


@dataclass
class BarcodeClassifierRL_Agent:
    confidence_score: float
