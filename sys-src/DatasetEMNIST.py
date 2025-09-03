from torch.utils.data import Dataset
from torchvision.transforms import v2
import random
from PrepareData import PrepareData


class OneOf:
    """
    Transformation, die zufällig eine von mehreren möglichen Bildtransformationen auswählt und anwendet.
    Ermöglicht die zufällige Auswahl von Augmentierungen mit optionalen Wahrscheinlichkeiten.
    """
    def __init__(self, transforms, probabilities=None):
        """
        :param transforms: Liste von Transformationsoperationen.
        :param probabilities: Wahrscheinlichkeiten für jede Transformation. Wenn None, werden alle gleich wahrscheinlich gewählt.
        :return: None
        """
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, image):
        """
        Wendet zufällig eine der angegebenen Transformationen auf das Bild an.
        :param image: Eingabebild.
        :return: Transformiertes Bild.
        """
        # Wähle eine Transformation basierend auf den Wahrscheinlichkeiten
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(image)


class DatasetEMNIST(Dataset):
    """
    Custom Dataset für EMNIST, das Label-Mapping und optionale Bildaugmentation unterstützt.
    Ermöglicht die gezielte Anwendung von Augmentierungen auf ausgewählte Bilder.
    """
    def __init__(self, dataset, valid_org_labels, lst_do_augmentation=None, probabilities=[1, 1, 1]):
        """
        Initialisiert das DatasetEMNIST.
        :param dataset: Ursprüngliches Dataset (z.B. Subset oder EMNIST).
        :param valid_org_labels: Liste der gültigen Original-Labels.
        :param lst_do_augmentation: Liste/Array, das angibt, ob für einen Index eine Augmentation angewendet werden soll.
        :param probabilities: Wahrscheinlichkeiten für die einzelnen Augmentierungen.
        :return: None
        """
        # Data already scaled from 0 to 1
        self.probabilities = probabilities
        self.dataset = dataset
        self.lst_do_augmentation = lst_do_augmentation
        self.valid_org_labels = valid_org_labels
        self.label_to_idx, self.idx_to_label_char = self.get_new_label_mapping(valid_org_labels)
        self.augmentations = [
            # v2.RandomRotation(degrees=(-20, 20)),
            #
            # # verschiebt die Zeichen
            # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            #
            # # sehr sehr gut, evtl. sogar besser als RandomResizedCrop
            # v2.RandomAffine(degrees=0, scale=(0.4, 1.2)),
            # # sehr gute kombi
            # v2.RandomAffine(degrees=0, scale=(0.4, 0.5), translate=(0.2, 0.3)),
            #
            # # sehr gut, evtl. 0.85 nehmen, 0.8 sollte aber auch klappen
            # v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            #
            # # für Robusteres Modell und Störezeichen
            # # v2.RandomErasing(p=0.5, scale=(0.01, 0.02), value=1),
            # # v2.RandomErasing(p=0.5, scale=(0.01, 0.02), value=0),
            # v2.GaussianBlur(kernel_size=(5, 5), sigma=(1, 10.5)),

            # 28x28
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.3)),
            v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1)),
            # v2.RandomAffine(degrees=0, scale=(0.7, 1.2), translate=(0.05, 0.15)),
            v2.RandomRotation(degrees=(-15, 15)),
        ]

        self.image_transforms = v2.Compose([
            OneOf(self.augmentations, probabilities=probabilities)
        ])


    def set_probabilities(self, probabilities):
        """
        Setzt die Wahrscheinlichkeiten für die Augmentierungen neu.
        :param probabilities: Neue Wahrscheinlichkeiten für die einzelnen Augmentierungen.
        :return: None
        """
        self.probabilities = probabilities
        self.image_transforms = v2.Compose([
            OneOf(self.augmentations, probabilities=probabilities)
        ])


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Gibt das Bild und das gemappte Label für den angegebenen Index zurück.
        Wendet ggf. eine Augmentierung auf das Bild an.
        :param idx: Index des gewünschten Elements.
        :return: (Bild, gemapptes Label)
        """
        img, label = self.dataset[idx]

        if self.lst_do_augmentation is not None and self.lst_do_augmentation[idx]:
            img = self.image_transforms(img)
        return img, self.label_to_idx[label]
    

    @staticmethod
    def get_new_label_mapping(valid_org_labels):
        """
        Erstellt ein Mapping von Original-Labels auf neue Indizes und von Indizes auf Zeichen.
        :param valid_org_labels: Liste der gültigen Original-Labels.
        :return: Tuple (label_to_idx, idx_to_label_char)
        """
        label_idx_to_char = PrepareData.get_char_from_label()
        label_to_idx = {orig_label: i for i, orig_label in enumerate(valid_org_labels)}  # 0 ... 48 --> 0 - 35
        idx_to_label_char = {i: label_idx_to_char[orig_label] for i, orig_label in
                             enumerate(valid_org_labels)}  # 0-35 --> 0-m

        return label_to_idx, idx_to_label_char


if __name__ == "__main__":
    valid_labels = list(range(0, 10)) + list(range(10, 23)) + list(range(36, 49))
    DatasetEMNIST.get_new_label_mapping(valid_labels)
