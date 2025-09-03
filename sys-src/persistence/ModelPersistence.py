import torch
import torch.nn as nn


class ModelPersistence:
    @staticmethod
    def save_model(model:nn.Module, optimizer, save_path_checkpoint_file):
        """
        Speichert den Zustand des Modells und des Optimizers in einer Datei.
        :param model: Das zu speichernde PyTorch-Modell.
        :param optimizer:  Der zugehörige Optimizer, dessen Zustand gespeichert werden soll.
        :param save_path_checkpoint_file: Der Pfad zur Datei, in der der Zustand gespeichert werden soll.
        :return: None
        """
        save_state = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(save_state, save_path_checkpoint_file)


    @staticmethod
    def load_model_from_checkpoint(model: nn.Module, save_path_checkpoint_file):
        """
        Lädt den Modellzustand aus einer Checkpoint-Datei und setzt den Zustand des übergebenen Modells zurück.
        :param model: Das Modell, dessen Zustand geladen werden soll.
        :param save_path_checkpoint_file: Der Pfad zur Checkpoint-Datei, aus der der Zustand geladen werden soll.
        :return: Das geladene Modell
        """
        if save_path_checkpoint_file is None or save_path_checkpoint_file == '':
            return

        state_dict = torch.load(save_path_checkpoint_file,
                                weights_only=False,
                                map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict["model"])

