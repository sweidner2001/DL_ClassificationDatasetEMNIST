import random
from collections import defaultdict
import torchvision.datasets as datasets
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import v2

class PrepareData:
    """
    Klasse zur Vorbereitung und Verarbeitung von Datensätzen für das Training und Testen von Modellen.
    Bietet Funktionen zum Filtern, Auswählen, Rebalancieren und Aufteilen von Datensätzen sowie zur Zuordnung von Labels zu Zeichen.
    """

    def __init__(self, train_dataset, test_dataset, valid_labels=None):
        """
        :param train_dataset: Trainingsdatensatz (z.B. EMNIST).
        :param test_dataset: Testdatensatz (z.B. EMNIST).
        :param valid_labels: Liste der gültigen Labels, die verwendet werden sollen. Wenn None, werden alle Labels aus dem Trainingsdatensatz verwendet.
        :return: None
        """
        self.valid_labels = valid_labels if valid_labels is not None else np.unique(train_dataset.targets)
        self.label_to_char = PrepareData.get_char_from_label()
        self.train_dataset, self.test_dataset = self._init_datasets(train_dataset, test_dataset)
        self.org_train_dataset = train_dataset
        self.org_test_dataset = test_dataset
        random.seed(42)


    def _init_datasets(self, train_dataset, test_dataset):
        """
        Initialisiert die Trainings- und Testdatensätze, ggf. gefiltert nach gültigen Labels.
        :param train_dataset: Ursprünglicher Trainingsdatensatz.
        :param test_dataset: Ursprünglicher Testdatensatz.
        :return: Gefilterter Trainings- und Testdatensatz.
        """
        if self.valid_labels is not None:
            # train_dataset_filtered = self.filter_dataset(train_dataset, self.valid_labels)
            # test_dataset_filtered = self.filter_dataset(test_dataset, self.valid_labels)
            # return train_dataset_filtered, test_dataset_filtered
            return train_dataset, test_dataset
        else:
            return train_dataset, test_dataset



    def filter_dataset(self, dataset, valid_labels):
        """
        Filtert das Dataset nach den benötigten Labels.
        :param dataset: Zu filterndes Dataset.
        :param valid_labels: Liste der gültigen Labels.
        :return: Gefiltertes Dataset als Subset.
        """
        filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in valid_labels]
        filtered_data = torch.utils.data.Subset(dataset, filtered_indices)
        return filtered_data


    def get_all_img_idx_pro_label(self, dataset):
        """
        Gibt pro Label-Klasse alle zugehörigen Bild-Indizes zurück.
        :param dataset: Dataset, das Labels enthält (muss .targets-Attribut haben oder durch Iteration zugänglich sein).
        :return: Dictionary {Label: [Indizes]}.
        """
        label_to_img_index = defaultdict(list)

        if isinstance(dataset, Subset):
            original_dataset = dataset.dataset
            targets = np.asarray(original_dataset.targets)[dataset.indices]

            for label in tqdm(self.valid_labels):
                indices = np.where(targets == label)[0]
                label_to_img_index[int(label)] = indices.tolist()

        elif hasattr(dataset, "targets"):
            for label in tqdm(self.valid_labels):
                indices = np.where(dataset.targets == label)[0]
                label_to_img_index[int(label)] = indices.tolist()
        else:
            label_to_img_index = defaultdict(list)
            for idx in tqdm(range(len(dataset)), desc="Images der Labels sammeln", unit="Label"):
                label = dataset[idx][1]
                label_to_img_index[int(label)].append(idx)

        return label_to_img_index


    # @staticmethod
    # def get_all_img_idx_pro_label(dataset):
    #     # Extrahiere die Labels als NumPy-Array
    #     labels = np.array([item[1] for item in tqdm(dataset)])
    #     unique_labels = np.unique(labels)
    #     return {int(label): np.where(labels == label)[0].tolist() for label in unique_labels}




    def get_data(self, num_samples_per_class_test=1000, num_samples_per_class_train=5000, rebalance_test_from_train=True, augm_train_if_neccessary=True, is_valid_dataset=True, valid_size=0.2):
        """
        Bereitet die Trainings-, Validierungs- und Testdatensätze vor, ggf. mit Rebalancierung und Augmentation.
        :param num_samples_per_class_test: Anzahl der Test-Samples pro Klasse.
        :param num_samples_per_class_train: Anzahl der Trainings-Samples pro Klasse.
        :param rebalance_test_from_train: Ob Testdaten ggf. aus Trainingsdaten ergänzt werden sollen.
        :param augm_train_if_neccessary: Ob Trainingsdaten bei Bedarf augmentiert werden sollen.
        :param is_valid_dataset: Ob ein Validierungsdatensatz erzeugt werden soll.
        :param valid_size: Anteil der Validierungsdaten.
        :return: Trainings-Subset, Augmentations-Flags, Validierungs-Subset, Validierungs-Flags, Testdatensatz.
        """
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset

        print("Länge:", len(train_dataset))
        if rebalance_test_from_train:
            train_dataset, test_dataset = self.rebalance_test_from_train(self.train_dataset, self.test_dataset,
                                                                         num_samples_per_class_test=num_samples_per_class_test,
                                                                         upper_num_samples_limit=True)

        print("Länge danch:", len(train_dataset))
        train_dataset, lst_do_augmentation = self.select_samples_train(train_dataset, num_samples_per_class_train, augm_train_if_neccessary=augm_train_if_neccessary)

        if is_valid_dataset:
            # Extrahiere die Daten- und Label-Listen
            data_indices = list(range(len(train_dataset)))

            # Train-Test Split mit sklearn
            train_indices, valid_indices, lst_do_augmentation_train, lst_train_do_augm_valid = train_test_split(
                data_indices, lst_do_augmentation, test_size=valid_size, random_state=42
            )
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            valid_subset = torch.utils.data.Subset(train_dataset, valid_indices)
            return train_subset, lst_do_augmentation_train, valid_subset, lst_train_do_augm_valid, test_dataset

        return train_dataset, lst_do_augmentation, None, None, test_dataset



    def rebalance_test_from_train(self, dataset_train, dataset_test, num_samples_per_class_test,
                                  upper_num_samples_limit=True):
        """
        Stellt sicher, dass jede Klasse im Testdatensatz die gewünschte Anzahl an Samples enthält.
        Ergänzt ggf. fehlende Testdaten aus dem Trainingsdatensatz und entfernt diese dort.
        :param dataset_train: Trainingsdatensatz.
        :param dataset_test: Testdatensatz.
        :param num_samples_per_class_test: Zielanzahl an Test-Samples pro Klasse.
        :param upper_num_samples_limit: Ob ein oberes Limit für Test-Samples pro Klasse gesetzt werden soll.
        :return: Angepasster Trainings- und Testdatensatz.
        """
        print("---rebalance_test_from_train()---")
        output_str = []
        final_test_orig_img_idx = []
        final_test_extra_img_idx = []
        final_train_img_idx = []


        # Bild-Indexe pro Label holen:
        train_img_idx_pro_label = self.get_all_img_idx_pro_label(dataset_train)
        test_img_idx_pro_label = self.get_all_img_idx_pro_label(dataset_test)

        for label in tqdm(sorted(self.valid_labels), desc="Prepare Test-Dataset", unit="label"):
            test_images_idx = test_img_idx_pro_label.get(label, []).copy()
            train_images_idx = train_img_idx_pro_label.get(label, []).copy()

            # get missing samples from train dataset
            needed = num_samples_per_class_test - len(test_images_idx)
            if needed > 0:
                if len(train_images_idx) < needed:
                    raise RuntimeError(f"Not enough Train-Samples for Label {self.label_to_char[label]}: needed: {needed}, available {len(train_images_idx)}.")

                # get extra test samples from train dataset and remove them from trian:
                extra_img_from_train = random.sample(train_images_idx, needed)
                train_images_idx = [i for i in train_images_idx if i not in extra_img_from_train]

                # test images:
                final_test_extra_img_idx.extend(extra_img_from_train)
                final_test_orig_img_idx.extend(test_images_idx)

                output_str.append(f"Klasse {self.label_to_char[label]}: Test {len(test_images_idx)} - Train {len(extra_img_from_train)}")

            # to many samples:
            elif upper_num_samples_limit and len(test_images_idx) > num_samples_per_class_test:
                # test images:
                keep_test_img_idx = random.sample(test_images_idx, num_samples_per_class_test)
                final_test_orig_img_idx.extend(keep_test_img_idx)

                output_str.append(f"Klasse {self.label_to_char[label]}: Test {len(keep_test_img_idx)} - original Test {len(test_images_idx)}")

            # train images
            final_train_img_idx.extend(train_images_idx)

        # new Subsets:
        new_train_ds = Subset(self.org_train_dataset, final_train_img_idx)
        test_subset_org = Subset(self.org_test_dataset, final_test_orig_img_idx)
        test_subset_from_train = Subset(self.org_train_dataset, final_test_extra_img_idx)
        new_test_ds = ConcatDataset([test_subset_org, test_subset_from_train])

        # Output number of images
        for s in output_str:
            print(s)

        return new_train_ds, new_test_ds



    @staticmethod
    def get_char_from_label(path='../data/EMNIST/raw/emnist-byclass-mapping.txt'):
        """
        Lädt die Zuordnung von Label-Indizes zu ASCII-Zeichen aus einer Mapping-Datei.
        :param path: Pfad zur Mapping-Datei.
        :return: Dictionary {Label-Index: Zeichen}.
        """
        # Label Zuordnung:
        label_idx_to_char = {}
        with open(path) as f:
            for line in f:
                idx, ascii_code = map(int, line.strip().split())
                label_idx_to_char[idx] = chr(ascii_code)

        return label_idx_to_char


    def select_samples_train(self, dataset_train, num_samples_per_class, augm_train_if_neccessary=True):
        """
        Wählt pro Klasse eine bestimmte Anzahl an Trainings-Samples aus und markiert ggf. zusätzliche für Augmentation.
        :param dataset_train: Trainingsdatensatz.
        :param num_samples_per_class: Zielanzahl an Trainings-Samples pro Klasse.
        :param augm_train_if_neccessary: Ob fehlende Samples durch Augmentation ergänzt werden sollen.
        :return: Neuer Trainings-Subset und Liste der Augmentations-Flags.
        """
        list_images = []
        list_augmentation_flags = []
        train_img_idx_pro_label = self.get_all_img_idx_pro_label(dataset_train)
        output_str = []


        for label in tqdm(sorted(train_img_idx_pro_label.keys()), desc="Prepare Train-Dataset", unit="label"):
            images_idx = train_img_idx_pro_label[label]

            # too many samples
            if len(images_idx) >= num_samples_per_class:
                chosen = random.sample(images_idx, num_samples_per_class)
                list_images.extend(chosen)
                list_augmentation_flags.extend(torch.zeros(num_samples_per_class, dtype=torch.bool))

                output_str.append(f"Label {self.label_to_char[label]}: Train {len(chosen)} orig {len(chosen)}")
            else:
                # too less samples
                list_images.extend(images_idx)
                list_augmentation_flags.extend(torch.zeros(len(images_idx), dtype=torch.bool))

                if augm_train_if_neccessary:
                    # Die Augmentations werden über alle Samples gleichmäßig ausgewählt.
                    # Wenn die aktuellen Samples nur 1/3 der gesamtzahl betragen, wird jedes sample mit Augmentation-Flag noch 2x hinzugefügt.
                    # Wenn die aktuellen Samples 1/4 sind, wird jedes Sample noch einmal mit Flag hinzugefügt und die restlichen zufällig gezogen.
                    augment_samples_needed = num_samples_per_class - len(images_idx)

                    output_str.append(f"Label {self.label_to_char[label]}: Train {len(images_idx)} augmentated {augment_samples_needed}")

                    while augment_samples_needed != 0:
                        add_nr_of_samples = min(augment_samples_needed, len(images_idx))

                        chosen = random.sample(images_idx, add_nr_of_samples)
                        list_images.extend(chosen)
                        list_augmentation_flags.extend(torch.ones(add_nr_of_samples, dtype=torch.bool))
                        augment_samples_needed -= add_nr_of_samples

            output_str.append(f"Trainingsdaten nach Label {self.label_to_char[label]}: {len(list_images)}")

        lst_img_indices = torch.tensor(list_images, dtype=torch.int)
        lst_do_augmentation = torch.tensor(list_augmentation_flags, dtype=torch.bool)
        new_train_ds = Subset(dataset_train, lst_img_indices)

        # Output number of images
        for s in output_str:
            print(s)


        return new_train_ds, lst_do_augmentation


    def check_number_of_samples(self, dataset):
        """
        Gibt für jedes Label die Anzahl der zugehörigen Samples im Datensatz aus.
        :param dataset: Zu überprüfender Datensatz.
        :return: None
        """
        label_to_img_index = defaultdict(list)
        for idx in tqdm(range(len(dataset)), desc="Images der Labels sammeln", unit="Label"):
            label = dataset[idx][1]
            label_to_img_index[int(label)].append(idx)

        for label in sorted(label_to_img_index.keys()):
            print(f"Klasse {self.label_to_char[label]}: {len(label_to_img_index[label])}")


    @staticmethod
    def get_normalization_train():
        """
        Gibt den Mittelwert und die Standardabweichung für die Normalisierung des Trainingsdatensatzes zurück.
        :return: Mittelwert und Standardabweichung (float oder Tensor).
        """

        return 0.1736, 0.3248

        # Datenvorbereitung ohne Normalisierung
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        # EMNIST-Dataset laden
        train_dataset = datasets.EMNIST(root='../data/', train=True, split="byclass", transform=transform,
                                        download=True)

        # Mittelwert und Standardabweichung berechnen
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=2000, shuffle=False)
        mean = 0.0
        std = 0.0
        n_samples = 0

        for data, _ in tqdm(loader, desc="Calc Std/Mean on training data"):
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            n_samples += batch_samples

        mean /= n_samples
        std /= n_samples
        # mean: tensor([0.1736]), std: tensor([0.3248])
        print(f"mean: {mean}, std: {std}")

        return mean, std




if __name__ == "__main__":
    print("Start")
    train_dataset = datasets.EMNIST(root='data/', train=True, split="byclass", transform=transforms.ToTensor(),
                                    download=True)
    test_dataset = datasets.EMNIST(root='data/', train=False, split="byclass", transform=transforms.ToTensor(),
                                   download=True)

    # relevante Labels:
    valid_labels = list(range(0, 10)) + list(range(10, 23)) + list(range(36, 49))

    # Erstelle die gewünschte Liste der Labels
    wanted_labels = list(range(0, 10)) + list(range(10, 23)) + list(range(36, 49))


    prepareData = PrepareData(train_dataset, test_dataset=test_dataset, valid_labels=valid_labels)
    train_dataset, lst_train_do_augm, test_dataset = prepareData.get_data()

    prepareData.check_number_of_samples(train_dataset)
    prepareData.check_number_of_samples(test_dataset)


