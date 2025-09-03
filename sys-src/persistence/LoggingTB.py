from torch.utils.tensorboard import SummaryWriter


class LoggingTB:

    ############################# Constructors: ###############################
    def __init__(self, save_path, filename_suffix:str=''):
        """
        :param save_path: Pfad zum Verzeichnis, in dem die TensorBoard-Logs gespeichert werden.
        :param filename_suffix: Optionaler Suffix für den Log-Dateinamen.
        """
        self.writer = SummaryWriter(log_dir=save_path, filename_suffix=filename_suffix)



    ############################# Methods: ###############################
    def log_dict(self, key_value_dict, step=None):
        """
        Loggt mehrere Schlüssel-Wert-Paare als Skalarwerte in TensorBoard.
        :param key_value_dict: Dictionary mit Schlüssel-Wert-Paaren (z.B. Metriken).
        :param step: Optionaler Schritt (z.B. Epoche oder Iteration), zu dem die Werte geloggt werden.
        :return: None
        """
        for key, value in key_value_dict.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()


    def log_value(self, key, value, step=None):
        """
        Loggt einen einzelnen Skalarwert in TensorBoard.
        :param key: Name des Wertes (z.B. 'loss' oder 'accuracy').
        :param value: Zu loggender Wert.
        :param step: Optionaler Schritt (z.B. Epoche oder Iteration), zu dem der Wert geloggt wird.
        :return: None
        """
        self.writer.add_scalar(key, value, step)

