import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import optuna
from torchvision.models import resnet18, ResNet18_Weights

# Eigene Imports:
from CombinedClassifier import CombinedClassifier
from CombinedClassifier2 import CombinedClassifier2
from ModelLogging import ModelLogging
from PrepareData import PrepareData
from DatasetEMNIST import DatasetEMNIST



class EMNISTModel:
    """
    Klasse zur Verwaltung des Trainings, Testens und der Auswertung von Modellen auf dem EMNIST-Datensatz.
    Beinhaltet Methoden zur Initialisierung, Trainings- und Testdurchführung, Logging, Datenaufbereitung und Modellverwaltung.
    """

    def __init__(self, experiment_name, validation_dataset_for_training=True):
        """
        :param experiment_name: Name des Experiments (für Logging und Checkpoints).
        :param validation_dataset_for_training: Ob ein Validierungsdatensatz für das Training verwendet werden soll.
        :return: None
        """
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_weights = ResNet18_Weights.IMAGENET1K_V1
        self.model_obj = resnet18
        self.df_best_metrics = None
        self.model = None   
        self.experiment_name = experiment_name

        self.path_tb_logging = "../Logging/TensorBoard/"
        self.path_checkpoints = "../Logging/Checkpoints/"
        self.path_cm_matrix = "../Logging/ConfusionMatrix/"

        self.relevant_labels = self.get_relevant_labels()
        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_train_test_dataset(validation=validation_dataset_for_training)



    ############################# Methods: ###############################

    #********************* Public methdos: ***********************
    def test(self, checkpoint_path=None, test_with_CombinedClassifier=False, config_combinedClassifier=None):
        """
        Testet das Modell auf dem Testdatensatz und gibt Metriken sowie die Konfusionsmatrix aus.
        :param checkpoint_path: Pfad zu einem gespeicherten Modell-Checkpoint.
        :param test_with_CombinedClassifier: Ob der CombinedClassifier verwendet werden soll.
        :param config_combinedClassifier: Konfiguration für den CombinedClassifier.
        :return: None
        """
        # init training:
        model = self._init_model_test(checkpoint_path, test_with_CombinedClassifier=test_with_CombinedClassifier, config_combinedClassifier=config_combinedClassifier)
        loss_fn = nn.CrossEntropyLoss()
        test_loader = self.get_dataloader(self.test_dataset)

        # testing:
        print("\n\n------ start model testing ------")
        df_metrics_testing, cm, test_loss = self._model_evaluation(model, loss_fn, test_loader)

        # Metrics
        print("Testing finished!")
        print(df_metrics_testing)
        print("Loss:", test_loss)

        # Confusion-Matrix visualisieren
        label_to_idx, idx_to_label_char = DatasetEMNIST.get_new_label_mapping(self.relevant_labels)
        plt.figure(figsize=(12, 8))
        class_names = [f"{idx_to_label_char[i]}" for i in range(36)]  # Beispielhafte Klassenlabels
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        os.makedirs(self.path_cm_matrix, exist_ok=True)
        plt.savefig(f"{self.path_cm_matrix}{self.experiment_name}.png")
        plt.show()




    def train(self, config, logging=False, trial=None):
        """
        Führt das Training des Modells durch.
        :param config: Dictionary mit Trainingsparametern und Modellkonfiguration.
        :param logging: Ob TensorBoard-Logging und Checkpointing aktiviert werden soll.
        :param trial: Optuna-Trial-Objekt für Hyperparameter-Tuning.
        :return: Beste Validierungsgenauigkeit (float)
        """

        print(config)

         # config = {
        #     'lr': 3.7046553541280545e-05,
        #     "weight_decay": 0.00132,
        #     'num_epochs': 20,
        #     'batch_size': 64,
        #     'only_train_classifier': False,
        #     'sched_patience': 4,
        #     'sched_factor': 0.5727064088839257,
        #     'sched_min_lr': 1e-05,
        #     'probabilities': [1, 1, 1],
        #     'train_with_CombinedClassifier': False,
        #     'config_CombinedClassifier': None
        # }

        # init logging:
        print("\n\n-------------- Start New Training: ---------------\n\n")
        model_logging_tb = None
        valid_loader = None
        if logging is True:
            model_logging_tb = ModelLogging(tb_path=f'{self.path_tb_logging}{self.experiment_name}',
                                            saving_path_checkpoint_file=f'{self.path_checkpoints}{self.experiment_name}')


        # init training:
        model = self._init_model_train(only_train_classifier=config['only_train_classifier'],
                                       train_with_CombinedClassifier=config['train_with_CombinedClassifier'],
                                       config_combinedClassifier=config['config_CombinedClassifier'])
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config['sched_patience'], factor=config['sched_factor'], min_lr=config['sched_min_lr'])

        # init data:
        self.train_dataset.set_probabilities(config['probabilities'])
        train_loader = self.get_dataloader(self.train_dataset, batch_size=config['batch_size'])
        best_valid_accuracy = 0
        if self.valid_dataset is not None:
            self.valid_dataset.set_probabilities(config['probabilities'])
            valid_loader = self.get_dataloader(self.valid_dataset, batch_size=config['batch_size'])

        # training:
        model.train()

        for epoch in range(config['num_epochs']):

            train_loss = 0

            #-------------------- Batch Loop: --------------------
            with tqdm(train_loader, unit="batch") as batch_iterator:
                batch_iterator.set_description(f"Epoch {epoch+1}/{config['num_epochs']}")
                get_absolute_step = lambda: (batch_idx + 1) + epoch * len(train_loader)

                for batch_idx, (X, y_true) in enumerate(batch_iterator):
                    # prediction:
                    X, y_true = X.to(self.device), y_true.to(self.device)
                    y_pred = model(X)

                    # loss: (CrossEntropyLoss expected Logits)
                    loss = loss_fn(y_pred, y_true)
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging loss:
                    if model_logging_tb is not None and (batch_idx + 1) % 10 == 0:
                        model_logging_tb.log_loss(batch_iterator, train_loss/10, optimizer, get_absolute_step())
                        train_loss = 0


            #-------------- am Ende von jeder Epoche validieren: ----------
            if valid_loader is not None:
                metrics_valid, cm, valid_loss = self._model_validation(model, loss_fn, valid_loader)
                scheduler.step(valid_loss)
                print(f"Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()}")


                if best_valid_accuracy <= metrics_valid['top1_score']:
                    best_valid_accuracy = metrics_valid['top1_score']

                # Optuna-Reporting und Pruning
                if trial is not None:
                    trial.report(metrics_valid['top1_score'], step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                # Logging:
                if model_logging_tb is not None:
                    self._logging_after_validation(model_logging_tb=model_logging_tb, step=epoch,
                                                   optimizer=optimizer, model=model,
                                                   df_current_metrics=metrics_valid)
            #----------------------------------------------------------------


        # save model
        print("Training finished!")
        self.model = model
        if model_logging_tb is not None:
            model_logging_tb.save_model(model=model, optimizer=optimizer, file_name="finalModel.pth")

        return best_valid_accuracy



    def get_train_test_dataset(self, validation=True):
        """
        Lädt und bereitet die Trainings-, Validierungs- und Testdatensätze vor.
        :param validation: Ob ein Validierungsdatensatz erzeugt werden soll.
        :return: Tuple (train_dataset, valid_dataset, test_dataset)
        """
        mean, std = PrepareData.get_normalization_train()


        # transforms_obj = self.model_weights.transforms()
        # print(f"mean: {transforms_obj.mean}, std: {transforms_obj.std}")

        # TODO: evtl. std und mean von EMNIST Datensatz nehmen
        prep = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),                           # [1,28,28], Werte in [0,1]
            # v2.Grayscale(num_output_channels=3),
            # v2.Resize((224, 224)),                   # [3,224,224]
            # v2.Normalize(
            #     mean=[mean, mean, mean],
            #     std=[std, std, std]
            # )
            v2.Normalize(
                mean=[mean],
                std=[std]
            )
        ])


        train_dataset = datasets.EMNIST(root='../data/', train=True, split="byclass", transform=prep,
                                        download=True)
        test_dataset = datasets.EMNIST(root='../data/', train=False, split="byclass", transform=prep,
                                       download=True)

        # relevante Labels / Daten:
        valid_labels = self.get_relevant_labels()
        prepareData = PrepareData(train_dataset, test_dataset=test_dataset, valid_labels=valid_labels)
        train_dataset, lst_train_do_augm_train, valid_dataset, lst_train_do_augm_valid, test_dataset = prepareData.get_data(num_samples_per_class_test=1000,
                                                                                                                          num_samples_per_class_train=5000,
                                                                                                                          is_valid_dataset=validation)
        train_dataset = DatasetEMNIST(train_dataset, self.relevant_labels, lst_do_augmentation=lst_train_do_augm_train)
        test_dataset = DatasetEMNIST(test_dataset, self.relevant_labels, lst_do_augmentation=None)
        if validation:
            valid_dataset = DatasetEMNIST(valid_dataset, self.relevant_labels, lst_do_augmentation=lst_train_do_augm_valid)

        return train_dataset, valid_dataset, test_dataset


    def get_dataloader(self, dataset, batch_size=32):
        """
        Erstellt einen DataLoader für das angegebene Dataset.
        :param dataset: Eingabedatensatz.
        :param batch_size: Batch-Größe.
        :return: DataLoader-Objekt.
        """
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=False)
        return dataloader


    @staticmethod
    def get_relevant_labels(print_labels: object = False) -> list[int]:
        """
        Gibt die Liste der relevanten Label-Indizes für das Training/Testen zurück.
        :param print_labels: Ob die Label-Charaktere ausgegeben werden sollen.
        :return: Liste der Label-Indizes.
        """
        valid_labels = list(range(0, 10)) + list(range(10, 23)) + list(range(36, 49))

        if print_labels:
            label_idx_to_char = PrepareData.get_char_from_label()
            for idx in [0, 9, 10, 22, 36, 48]:
                print(idx, '→', label_idx_to_char[idx])

        return valid_labels




    #********************* Private methdos: ***********************
    def _model_validation(self, model, loss_fn, valid_loader:DataLoader):
        """
        Führt die Validierung des Modells auf dem Validierungsdatensatz durch.
        :param model: Zu validierendes Modell.
        :param loss_fn: Loss-Funktion.
        :param valid_loader: DataLoader für den Validierungsdatensatz.
        :return: Tuple (Metriken, Konfusionsmatrix, Loss)
        """
        print("\nModel validation...")
        metrics_testing, cm, valid_loss = self._model_evaluation(model, loss_fn, valid_loader)
        print(metrics_testing)
        print("valid_loss: ", valid_loss)
        model.train()
        return metrics_testing, cm, valid_loss


    def _model_evaluation(self, model, loss_fn, valid_loader):
        """
        Führt die Auswertung des Modells auf einem Datensatz durch und berechnet Metriken.
        :param model: Modell zur Auswertung.
        :param loss_fn: Loss-Funktion.
        :param valid_loader: DataLoader für den Datensatz.
        :return: Tuple (Metriken, Konfusionsmatrix, Loss)
        """
        # every new key will be init with a list:
        model.eval()
        y_true_all = []
        y_pred_all = []
        valid_loss = 0

        with torch.no_grad():
            # ------------ model testing: -----------
            top_5_accuracy = MulticlassAccuracy(num_classes=36, top_k=5).to(self.device)
            top_3_accuracy = MulticlassAccuracy(num_classes=36, top_k=3).to(self.device)
            top_1_accuracy = MulticlassAccuracy(num_classes=36, top_k=1).to(self.device)

            with tqdm(valid_loader, unit="batch") as valid_iterator:
                for valid_batch, (X_valid, y_valid) in enumerate(valid_iterator):
                    # prediction:
                    X, y_true = X_valid.to(self.device), y_valid.to(self.device)
                    y_pred = model(X)

                    loss = loss_fn(y_pred, y_true)
                    valid_loss += loss.detach().cpu().numpy()

                    # predicted_class = torch.argmax(y_pred, 1)
                    # Vorhersagen und Labels sammeln
                    y_true_all.extend(y_true.cpu().numpy())
                    y_pred_all.extend(torch.argmax(y_pred, 1).cpu().numpy())  # Argmax für die Vorhersageklasse

                    top_5_accuracy.update(y_pred, y_true)
                    top_3_accuracy.update(y_pred, y_true)
                    top_1_accuracy.update(y_pred, y_true)

                valid_loss /= (valid_batch + 1)

            final_top5_score = top_5_accuracy.compute().item()
            final_top3_score = top_3_accuracy.compute().item()
            final_top1_score = top_1_accuracy.compute().item()

        metrics = pd.Series(data={
            f"top1_score": final_top1_score,
            f"top3_score": final_top3_score,
            f"top5_score": final_top5_score,
        })
        cm = confusion_matrix(y_true_all, y_pred_all)

        return metrics, cm, valid_loss




    # ~~~~~~~~~~~~~~~~~~~ Logging ~~~~~~~~~~~~~~~~~~~~~~
    def _logging_after_validation(self, model_logging_tb: ModelLogging, step: int, optimizer, model, df_current_metrics):
        """
        Führt das Logging nach einer Validierungsepoche durch (Metriken, Bestwerte, Modell speichern).
        :param model_logging_tb: ModelLogging-Objekt für TensorBoard und Checkpoints.
        :param step: Aktuelle Epoche.
        :param optimizer: Optimizer.
        :param model: Modell.
        :param df_current_metrics: Aktuelle Metriken als pandas.Series.
        :return: None
        """
        # log current metrics:
        model_logging_tb.log_dict(df_current_metrics.to_dict(), step)

        # # log steps of the best metrics:
        self.df_best_metrics = model_logging_tb.log_best_metrics(df_best_metrics=self.df_best_metrics,
                                                                 current_metrics=df_current_metrics,
                                                                 step=step,
                                                                 model=model,
                                                                 optimizer=optimizer)
        print(df_current_metrics)
        print('\n' + self.df_best_metrics.to_string())






    #~~~~~~~~~~~~~~~~~~~ Init ~~~~~~~~~~~~~~~~~~~~~
    def _init_model_test(self, checkpoint_path=None, test_with_CombinedClassifier=False, config_combinedClassifier=None):
        """
        Initialisiert das Modell für den Testmodus und lädt ggf. einen Checkpoint.
        :param checkpoint_path: Pfad zu einem gespeicherten Modell-Checkpoint.
        :param test_with_CombinedClassifier: Ob der CombinedClassifier verwendet werden soll.
        :param config_combinedClassifier: Konfiguration für den CombinedClassifier.
        :return: Initialisiertes Modell.
        """
        model = self.model
        if model is None:
            model = self._init_model_train(train_with_CombinedClassifier=test_with_CombinedClassifier, config_combinedClassifier=config_combinedClassifier)

            if checkpoint_path is not None:
                if not os.path.isfile(checkpoint_path):
                    raise Exception("Their exists no checkpoint file with the path:", checkpoint_path)

                ModelLogging.load_model_from_checkpoint(model, checkpoint_path)


        model.to(self.device)
        model.eval()
        return model




    def _init_model_train(self, only_train_classifier=False, train_with_CombinedClassifier=False, config_combinedClassifier=None):
        """
        Initialisiert das Modell für das Training, ggf. mit CombinedClassifier.
        :param only_train_classifier: Ob nur der Klassifikator trainiert werden soll (Feature-Extraktor einfrieren).
        :param train_with_CombinedClassifier: Ob der CombinedClassifier verwendet werden soll.
        :param config_combinedClassifier: Konfiguration für den CombinedClassifier.
        :return: Initialisiertes Modell.
        """
        model = self.model_obj(weights=self.model_weights)

        # Alle Schichten einfrieren
        if only_train_classifier:
            for param in model.parameters():
                param.requires_grad = False


        # Erster Convolutional Layer anpassen
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # own classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.relevant_labels))



        if train_with_CombinedClassifier:
            model = CombinedClassifier(tm1_model=model, config=config_combinedClassifier,
                                          num_out_classes_tm2=3, tm2_mapping_idx=[[0, 9], [10, 22], [23, 35]])

            # model.fc = CombinedClassifier2(in_features_tm=in_features, config=config_combinedClassifier, num_out_classes_tm1=len(self.relevant_labels))
        model.to(self.device)
        return model



class HyperparameterTuning:
    """
    Klasse zur Durchführung und Auswertung von Hyperparameter-Tuning mit Optuna für das EMNIST-Modell.
    Beinhaltet Methoden zum Starten des Tuning-Prozesses, zur Auswertung der besten Parameter und Trials sowie zur Übersicht aller Studien.
    """

    STORAGE_PATH = "../Logging/Optuna"

    def __init__(self, experiment_name="HyperparameterTuning_1", storage_path=STORAGE_PATH):
        """
        :param experiment_name: Name des Experiments/der Studie.
        :param storage_path: Pfad zum Speicherort der Optuna-Datenbank.
        :return: None
        """
        self.experiment_name = experiment_name
        self.emnist = None
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def startTuning(self, n_trails):
        """
        Startet das Hyperparameter-Tuning mit Optuna.
        :param n_trails: Anzahl der Trials (Versuche) für das Tuning.
        :return: None
        """
        self.emnist = EMNISTModel(experiment_name=self.experiment_name)
        study = optuna.create_study(study_name=self.experiment_name,
                                    storage=f'sqlite:///{self.storage_path}/optuna.db',
                                    load_if_exists=True,
                                    direction="maximize",
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, interval_steps=3))  # Wir wollen den höchsten Reward
                                    # pruner=optuna.pruners.ThresholdPruner(upper=0.85, n_warmup_steps=5)
        study.optimize(self.objective, n_trials=n_trails)

        print("Beste Hyperparameter:", study.best_params)
        print("Beste Leistung (Maximale top1_prediction):", study.best_value)




    def objective(self, trial):
        """
        Definiert die Ziel-Funktion für Optuna, die für jeden Trial aufgerufen wird.
        :param trial: Optuna-Trial-Objekt.
        :return: Beste Validierungsgenauigkeit (float)
        """
        # Hyperparameter-Vorschläge
        # config_CombinedClassifier = {
        #         'linear_out_1': trial.suggest_categorical("linear_out_1", [128, 256, 512]),
        #         'linear_out_2': trial.suggest_categorical("linear_out_2", [64, 128, 256]),
        #         'dropout': trial.suggest_float("dropout", 0.0, 0.4)
        #     }
        # config_CombinedClassifier = {
        #     'conv2d_out_1': trial.suggest_categorical("conv2d_out_1", [32, 64]),
        #     'conv2d_kernel_size_1': trial.suggest_categorical("conv2d_kernel_size_1", [3, 5]),
        #     'conv2d_out_2': trial.suggest_categorical("conv2d_out_2", [32, 64]),
        #     'conv2d_kernel_size_2': trial.suggest_categorical("conv2d_kernel_size_2", [3, 5]),
        #     'conv2d_out_3': trial.suggest_categorical("conv2d_out_3", [32, 64, 128]),
        #     'conv2d_kernel_size_3': trial.suggest_categorical("conv2d_kernel_size_3", [3, 5]),
        #     'linear_out_1': trial.suggest_categorical("linear_out_1", [128, 256, 512]),
        #     'linear_out_2': trial.suggest_categorical("linear_out_2", [32, 64, 128]),
        #     'dropout': trial.suggest_float("dropout", 0.1, 0.4),
        # }
        #
        # config = {
        #     "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        #     "weight_decay": trial.suggest_float("weight_decay", 0.0002, 0.01, log=True),
        #     "num_epochs": 25,
        #     "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        #     "only_train_classifier": False,
        #     "sched_patience": 5,
        #     "sched_factor": trial.suggest_float("sched_factor", 0.4, 0.7),
        #     "sched_min_lr": 1e-6,
        #     'train_with_CombinedClassifier': True,
        #     'probabilities': [0.5, 1, 1],
        #     'config_CombinedClassifier': config_CombinedClassifier
        # }
        trial.set_user_attr("train_with_CombinedClassifier", True)
        trial.set_user_attr("only_train_classifier", False)
        trial.set_user_attr("num_epochs", 25)
        trial.set_user_attr("sched_min_lr", 1e-6)
        trial.set_user_attr("sched_patience", 5)
        trial.set_user_attr("probabilities", [0.5, 1, 1])

        config_CombinedClassifier = {
            'conv2d_out_1': 32,
            'conv2d_kernel_size_1': 3,
            'conv2d_out_2': 64,
            'conv2d_kernel_size_2': 5,
            'conv2d_out_3': 64,
            'conv2d_kernel_size_3': 5,
            'linear_out_1': 128,
            'linear_out_2': 128,
            'dropout': 0.20,
        }
        config = {
            "lr": 3.3123809201322046e-05,
            "weight_decay": 0.000460,
            "num_epochs": 25,
            "batch_size": 32,
            "only_train_classifier": False,
            "sched_patience": 5,
            "sched_factor": 0.448307,
            "sched_min_lr": 0.000001,
            'train_with_CombinedClassifier': True,
            'probabilities': [0.5, 1, 1],
            'config_CombinedClassifier': config_CombinedClassifier
        }

        # config = {
        #     'lr': 3.7046553541280545e-05,
        #     'num_epochs': trial.suggest_int("sched_patience", 20, 50),
        #     'batch_size': 64,
        #     'only_train_classifier': False,
        #     'sched_patience': 14,
        #     'sched_factor': 0.266912507119884,
        #     'sched_min_lr': 1e-05,
        #     'probabilities': [0.5735191209078359, 0.9150760423853747, 1],
        #     'train_with_CombinedClassifier': False,
        #     'config_CombinedClassifier': None
        # }


        best_valid_accuracy = self.emnist.train(config, logging=False, trial=trial)
        return best_valid_accuracy

    def test(self):
        self.emnist.test()

    def get_best_params(self, experiment_name):
        """
        Gibt die besten Hyperparameter und Details des besten Trials für eine Studie aus.
        :param experiment_name: Name der Optuna-Studie.
        :return: Dictionary der besten Hyperparameter.
        """
        study = optuna.load_study(
            study_name=experiment_name,
            storage=f'sqlite:///{self.storage_path}/optuna.db',
        )

        # 2) Beste Params
        print("Beste Hyperparameter:", study.best_params)

        # 3) Details des besten Trials
        best = study.best_trial
        print(f"Bestes Trial #{best.number}:")
        print("  Value:", best.value)
        print("  Params:")
        for key, val in best.params.items():
            print(f"    {key}: {val}")


        return study.best_params


    def get_best_trials(self, experiment_name):
        """
        Gibt die Top-Trials einer Studie sortiert nach Leistung aus.
        :param experiment_name: Name der Optuna-Studie.
        :return: None
        """
        study = optuna.load_study(
            study_name=experiment_name,
            storage=f'sqlite:///{self.storage_path}/optuna.db',
        )

        # Trials als DataFrame exportieren und nach Leistung sortieren
        df = study.trials_dataframe()
        df_sorted = df.sort_values("value", ascending=False)  # Für Maximierung ("value" = Ihre Metrik)

        # Top 10 Trials anzeigen
        top_10_trials = df_sorted.head(10)
        # Iteriere über die Zeilen des DataFrames
        for _, row in top_10_trials.iterrows():
            print(row)
            print('\n\n')

        # fig = plot_param_importances(study)
        # fig.show()


    def get_all_studies(self):
        """
        Gibt eine Übersicht aller Optuna-Studien im angegebenen Speicherpfad aus.
        :return: None
        """
        # Alle Studien abrufen
        study_summaries = optuna.get_all_study_summaries(storage=f'sqlite:///{self.storage_path}/optuna.db')

        # Studien-Namen und Metadaten anzeigen
        for study in study_summaries:
            print(f"Study-ID: {study._study_id}")
            print(f"Name: {study.study_name}")
            print(f"Richtung: {study.direction}")  # 'maximize' oder 'minimize'
            print(f"Anzahl Trials: {study.n_trials}")
            print("-" * 50)






if __name__ == "__main__":

    print("Start")
    # HyperparameterTuning(experiment_name="Test_HyperparameterTuningT").startTuning(n_trails=2)
    # HyperparameterTuning(storage_path="../Logging_013/Logging_cc2/Optuna").get_all_studies()
    # best_params = HyperparameterTuning(storage_path="../Logging_013/Logging_cc2/Optuna").get_best_params(experiment_name="HyperparameterTuning_1")
    # best_trials = HyperparameterTuning(storage_path="../Logging_013/Logging_cc2/Optuna").get_best_trials(experiment_name="HyperparameterTuning_1")
    # print(best_params)

    # config = best_params

    config_CombinedClassifier = {
        'linear_out_1': 256,
        'linear_out_2': 256,
        'dropout': 0.18,
    }

    config_CombinedClassifier = {
        'conv2d_out_1': 16,
        'conv2d_kernel_size_1': 3,
        'conv2d_out_2': 32,
        'conv2d_kernel_size_2': 3,
        'conv2d_out_3': 32,
        'conv2d_kernel_size_3': 3,
        'linear_out_1': 256,
        'linear_out_2': 64,
        'dropout': 0.3,
    }
    config = {
        'lr': 3.7046553541280545e-05,
        "weight_decay": 0.00132,
        'num_epochs': 20,
        'batch_size': 64,
        'only_train_classifier': False,
        'sched_patience': 4,
        'sched_factor': 0.5727064088839257,
        'sched_min_lr': 1e-05,
        'probabilities': [1, 1, 1],
        'train_with_CombinedClassifier': True,
        'config_CombinedClassifier': config_CombinedClassifier
    }

    emnist = EMNISTModel(experiment_name="test", validation_dataset_for_training=False)
    emnist.train(config=config, logging=True)
    emnist.test(test_with_CombinedClassifier=config['train_with_CombinedClassifier'])


