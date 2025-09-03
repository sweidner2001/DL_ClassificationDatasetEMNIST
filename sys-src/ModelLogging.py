import os
from persistence.LoggingTB import LoggingTB
import pandas as pd
from persistence.ModelPersistence import ModelPersistence
import torch.nn as nn


class ModelLogging:
    """
    Klasse zur Verwaltung des Loggings von Trainingsmetriken und zur Modellpersistenz.
    Ermöglicht das Loggen von Metriken in TensorBoard sowie das Speichern und Laden von Modell-Checkpoints.
    """

    ############################# Constructors: ###############################
    def __init__(self, tb_path, saving_path_checkpoint_file):
        """
        :param tb_path: Pfad zum Verzeichnis für TensorBoard-Logs.
        :param saving_path_checkpoint_file: Pfad zum Verzeichnis, in dem Modell-Checkpoints gespeichert werden.
        :return: None
        """
        self.tb_log = LoggingTB(save_path=tb_path)
        self.saving_path_checkpoint_file = saving_path_checkpoint_file




    ############################# Methods: ###############################
    def log_dict(self, key_value_dict, step=None):
        """
        Loggt mehrere Schlüssel-Wert-Paare als Skalarwerte in TensorBoard.
        :param key_value_dict: Dictionary mit Metriken (z.B. Loss, Accuracy).
        :param step: Optionaler Schritt (z.B. Epoche oder Iteration), zu dem die Werte geloggt werden.
        :return: None
        """
        self.tb_log.log_dict(key_value_dict, step)


    def log_loss(self, batch_iterator, loss_avg, optimizer, step):
        """
        Loggt den aktuellen Loss-Wert und die Lernrate in TensorBoard und zeigt sie im Batch-Iterator an.
        :param batch_iterator: Iterator für die aktuelle Trainings-Batch (z.B. tqdm).
        :param loss_avg: Durchschnittlicher Loss-Wert der aktuellen Batch.
        :param optimizer: Optimizer, um die aktuelle Lernrate auszulesen.
        :param step: Schritt (z.B. Iteration oder Epoche), zu dem der Wert geloggt wird.
        :return: None
        """
        lr = optimizer.param_groups[0]['lr']
        self.tb_log.log_value("Loss/train", loss_avg, step)
        batch_iterator.set_postfix(lr=f'{lr:.6f}', loss=f'{loss_avg:.4f}')



    def log_best_metrics(self, df_best_metrics, step: int, current_metrics, model, optimizer):
        """
        Überwacht und loggt die besten erreichten Metriken. Speichert das Modell, wenn eine neue Bestleistung erreicht wird.
        :param df_best_metrics: DataFrame mit den bisher besten Metriken und den zugehörigen Schritten.
        :param step: Aktueller Schritt (z.B. Epoche).
        :param current_metrics: Aktuelle Metriken als pandas.Series.
        :param model: Das zu speichernde Modell.
        :param optimizer: Der zugehörige Optimizer.
        :return: Aktualisierter DataFrame mit den besten Metriken.
        """
        if df_best_metrics is None:
            df_best_metrics = pd.DataFrame(data={'best_value': current_metrics.values.copy(), 'on_step': step},
                                           index=current_metrics.index.copy())

        for metric_name in current_metrics.index:
            if current_metrics[metric_name] >= df_best_metrics.at[metric_name, 'best_value']:
                # TensorBoard: log best metric step
                self.tb_log.log_value(key=f'Steps - Best {metric_name}', value=current_metrics[metric_name], step=step)
                df_best_metrics.at[metric_name, 'best_value'] = current_metrics[metric_name]
                df_best_metrics.at[metric_name, 'on_step'] = step

                self.save_model(model, optimizer, file_name=f'best_{metric_name}.pth')

        return df_best_metrics


    def save_model(self, model, optimizer, file_name):
        """
        Speichert den Zustand des Modells und des Optimizers als Checkpoint-Datei.
        :param model: Das zu speichernde PyTorch-Modell.
        :param optimizer: Der zugehörige Optimizer.
        :param file_name: Dateiname für den Checkpoint.
        :return: None
        """
        os.makedirs(self.saving_path_checkpoint_file, exist_ok=True)
        ModelPersistence.save_model(model, optimizer, f'{self.saving_path_checkpoint_file}/{file_name}')


    @staticmethod
    def load_model_from_checkpoint(model: nn.Module, save_path_checkpoint_file):
        """
        Lädt den Modellzustand aus einer Checkpoint-Datei und setzt den Zustand des übergebenen Modells zurück.
        :param model: Das Modell, dessen Zustand geladen werden soll.
        :param save_path_checkpoint_file: Der Pfad zur Checkpoint-Datei, aus der der Zustand geladen werden soll.
        :return: Das geladene Modell
        """
        return ModelPersistence.load_model_from_checkpoint(model, save_path_checkpoint_file)
