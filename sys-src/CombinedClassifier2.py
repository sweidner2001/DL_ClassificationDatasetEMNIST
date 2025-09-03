import torch.nn.functional as F
import torch.nn as nn
import torch


class CombinedClassifier2(nn.Module):
    """
    Kombinierter Klassifikator mit zwei Modulen:
    - TM1: Klassifiziert direkt in feingranulare Klassen (z.B. 36 Klassen).
    - TM2: Gruppiert die feingranularen Klassen in Oberklassen (z.B. 3 Gruppen) und gewichtet die TM1-Ausgaben entsprechend.
    Die finale Vorhersage kombiniert die Wahrscheinlichkeiten beider Module anhand eines Index-Mappings.
    """
    def __init__(self, in_features_tm : int, config, num_out_classes_tm1=36, num_out_classes_tm2=3, tm2_mapping_idx=[[0, 9], [10, 22],[23, 35]]):
        """
        :param in_features_tm: Anzahl der Eingangsfeatures.
        :param config: Dictionary mit Layergrößen und Dropout-Rate für TM2.
        :param num_out_classes_tm1: Anzahl der Ausgabeklassen für TM1.
        :param num_out_classes_tm2: Anzahl der Ausgabeklassen für TM2 (Oberklassen).
        :param tm2_mapping_idx: Liste von [start_idx, end_idx] für die Zuordnung der TM2-Klassen zu TM1-Klassen.
        :return: None
        """
        super(CombinedClassifier2, self).__init__()

        # config = {
        #     'linear_out_1': 256,
        #     'linear_out_2': 128,
        #     'dropout': 0.3,
        # }

        # TM1: CNN für 36 Klassen
        self.tm2_mapping_idx = tm2_mapping_idx
        self.tm1 = nn.Linear(in_features_tm, num_out_classes_tm1)

        # TM2: Neues Modul
        self.tm2 = nn.Sequential(
            nn.Linear(in_features_tm, config['linear_out_1']),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['linear_out_1'], config['linear_out_2']),
            nn.ReLU(),
            nn.Linear(config['linear_out_2'], num_out_classes_tm2)
        )




    def forward(self, x):
        """
        Führt die Vorwärtsberechnung durch und kombiniert die Wahrscheinlichkeiten von TM1 und TM2.
        :param x: Eingabefeatures (Tensor).
        :return: Kombinierte Wahrscheinlichkeiten für alle feingranularen Klassen (Tensor).
        """
        
        # TM1 Vorhersagen
        logits_tm1 = self.tm1(x)
        probs_tm1 = F.softmax(logits_tm1, dim=1)

        # TM2 Vorhersagen
        logits_tm2 = self.tm2(x)
        probs_tm2 = F.softmax(logits_tm2, dim=1)

        # Initialisiere ein leeres Tensor für die kombinierten Wahrscheinlichkeiten
        combined_probs = torch.zeros_like(probs_tm1)

        # Kombiniere die Wahrscheinlichkeiten basierend auf den Mapping-Indexen
        for tm2_class, (start_idx, end_idx) in enumerate(self.tm2_mapping_idx):

            # Multiplizieren mit der entsprechenden Wahrscheinlichkeit aus TM2
            combined_probs[:, start_idx:end_idx + 1] = probs_tm1[:, start_idx:end_idx + 1] * probs_tm2[:, tm2_class].unsqueeze(1)

        return combined_probs

