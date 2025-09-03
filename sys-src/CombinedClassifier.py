import torch.nn.functional as F
import torch.nn as nn
import torch


class CombinedClassifier(nn.Module):
    """
    Kombinierter Klassifikator mit zwei Modulen:
    - TM1: Ein beliebiges Modell für feingranulare Klassifikation (z.B. 36 Klassen).
    - TM2: Eigenes CNN-Modul, das die feingranularen Klassen in Oberklassen (z.B. 3 Gruppen) gruppiert.
    Die finale Vorhersage kombiniert die Wahrscheinlichkeiten beider Module anhand eines Index-Mappings.
    """
    def __init__(self, tm1_model, config: dict, num_out_classes_tm2=3, tm2_mapping_idx=[[0, 9], [10, 22], [23, 35]]):
        """
        Initialisiert den CombinedClassifier.
        :param tm1_model: Vorgegebenes Modell für die feingranulare Klassifikation (TM1).
        :param config: Dictionary mit Layergrößen, Kernelgrößen und Dropout-Rate für TM2.
        :param num_out_classes_tm2: Anzahl der Ausgabeklassen für TM2 (Oberklassen).
        :param tm2_mapping_idx: Liste von [start_idx, end_idx] für die Zuordnung der TM2-Klassen zu TM1-Klassen.
        :return: None
        """
        # config = {
        #     'conv2d_out_1': 16,
        #     'conv2d_kernel_size_1': 3,
        #     'conv2d_out_2': 32,
        #     'conv2d_kernel_size_2': 3,
        #     'conv2d_out_3': 32,
        #     'conv2d_kernel_size_3': 3,
        #     'linear_out_1': 256,
        #     'linear_out_2': 64,
        #     'dropout': 0.3,
        # }
        super(CombinedClassifier, self).__init__()

        self.tm1_model = tm1_model
        self.tm2_mapping_idx = tm2_mapping_idx

        padding1 = int((config['conv2d_kernel_size_1']-1)/2)
        padding2 = int((config['conv2d_kernel_size_2']-1)/2)
        padding3 = int((config['conv2d_kernel_size_3']-1)/2)

        self.tm2 = nn.Sequential(
            # Block 1
            nn.Conv2d(1, config['conv2d_out_1'], kernel_size=config['conv2d_kernel_size_1'], stride=1, padding=padding1),  # (16, 28, 28)
            nn.BatchNorm2d(config['conv2d_out_1']),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 14, 14)

            # Block 2
            nn.Conv2d(config['conv2d_out_1'], config['conv2d_out_2'], kernel_size=config['conv2d_kernel_size_2'], stride=1, padding=padding2),  # (32, 14, 14)
            nn.BatchNorm2d(config['conv2d_out_2']),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(config['conv2d_out_2'], config['conv2d_out_3'], kernel_size=config['conv2d_kernel_size_3'], stride=1, padding=padding3),  # (32, 14, 14)
            nn.BatchNorm2d(config['conv2d_out_3']),
            nn.ReLU(),
            nn.MaxPool2d(2),                                                    # (64, 7, 7)

            nn.Flatten(),
            nn.Linear(config['conv2d_out_3'] * 7 * 7, config['linear_out_1']),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['linear_out_1'], config['linear_out_2']),
            nn.ReLU(),
            nn.Linear(config['linear_out_2'], num_out_classes_tm2)                 # 3 Klassen
        )



    def forward(self, x):
        """
        Führt die Vorwärtsberechnung durch und kombiniert die Wahrscheinlichkeiten von TM1 und TM2.
        :param x: Eingabebild (Tensor).
        :return: Kombinierte Wahrscheinlichkeiten für alle feingranularen Klassen (Tensor).
        """
        # TM1 Vorhersagen
        logits_tm1 = self.tm1_model(x)
        probs_tm1 = F.softmax(logits_tm1, dim=1)

        # TM2 Vorhersagen
        logits_tm2 = self.tm2(x)
        probs_tm2 = F.softmax(logits_tm2, dim=1)

        # Initialisiere ein leeres Tensor für die kombinierten Wahrscheinlichkeiten
        combined_probs = torch.zeros_like(probs_tm1)

        # # Kombiniere die Wahrscheinlichkeiten basierend auf den Mapping-Indexen
        # for tm2_class, (start_idx, end_idx) in enumerate(self.tm2_mapping_idx):
        #
        #     # Multiplizieren mit der entsprechenden Wahrscheinlichkeit aus TM2
        #     combined_probs[:, start_idx:end_idx + 1] = probs_tm1[:, start_idx:end_idx + 1] * probs_tm2[:, tm2_class].unsqueeze(1)

        # Vor der Schleife:
        mask = torch.zeros((probs_tm1.size(1), probs_tm2.size(1)), device=x.device)
        for tm2_class, (start_idx, end_idx) in enumerate(self.tm2_mapping_idx):
            mask[start_idx:end_idx + 1, tm2_class] = 1

        combined_probs = (probs_tm1.unsqueeze(2) * probs_tm2.unsqueeze(1) * mask.unsqueeze(0)).sum(dim=2)

        return combined_probs
