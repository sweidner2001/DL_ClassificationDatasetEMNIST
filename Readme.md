

# Projektstruktur
- ***[/data]()***: Hier befindet sich der EMNIST Datensatz
- ***[/Logging]()***: Protokolle des Modeltrainings und der Hyperparameter Optimierung
  - ***[/Checkpoints]()***: Abgespeicherte Modelle
  - ***[/ConfusionMatrix]()***: Confusion Matrizen der Finalen Modelle
  - ***[/ConsoleOutput]()***: Konsolen-Output zu den Trainingsprozessen der finalen Modelle - Hier kann die Leistung der Modelle abgelen werden.
  - ***[/Opuna]()***: Datenbanken zu Hyperparameter-Optimierungen aller drei Modelle
    - ***[/Ergebnisse_HT]()***: Ergebnisse der Hyperparameter-Optimierung zur Ansicht
  - ***[/TensorBoard]()***: TensorBoard-Protokolle zu den Trainingsprozessen der finalen Modelle
- **[/sys-doc]()**: Projektdokumentation und Aufgabenstellung
- **[/sys-src]()**: Quellcode

---

## Infos zum Quellcode

Im Ordner **/sys-src** befindet sich der gesamte Quellcode des Projekts. Die wichtigsten Dateien und deren zentrale Klassen sind:

### **main.py**
- **EMNISTModel**: Zentrale Klasse zur Steuerung des Trainings, Testens und der Auswertung von Modellen auf dem EMNIST-Datensatz. Sie kapselt die gesamte Pipeline von der Datenaufbereitung über das Modelltraining bis zur Auswertung und Logging.
- **HyperparameterTuning**: Klasse zur Durchführung und Auswertung von Hyperparameter-Tuning mit Optuna. Sie nutzt EMNISTModel für die Trainings- und Evaluationsschritte.

### **PrepareData.py**
- **PrepareData**: Klasse zur Vorbereitung und Verarbeitung der EMNIST-Daten. Sie bietet Methoden zum Filtern, Auswählen, Rebalancieren und Aufteilen der Datensätze sowie zur Label-Zuordnung. Sie wird von EMNISTModel zur Datenvorbereitung verwendet.

### **DatasetEMNIST.py**
- **DatasetEMNIST**: Custom Dataset für EMNIST, das Label-Mapping und gezielte Bildaugmentation unterstützt. Wird von EMNISTModel nach der Datenaufbereitung genutzt.
- **OneOf**: Hilfsklasse für zufällige Auswahl und Anwendung von Bildtransformationen (Augmentierungen).

### **ModelLogging.py**
- **ModelLogging**: Klasse zur Verwaltung des Loggings von Trainingsmetriken (z.B. für TensorBoard) und zur Modellpersistenz (Speichern/Laden von Checkpoints). Wird im Training von EMNISTModel verwendet.
- Nutzt intern **LoggingTB** (siehe unten) und **ModelPersistence**.

### **persistence/LoggingTB.py**
- **LoggingTB**: Kapselt das Logging von Metriken in TensorBoard. Wird von ModelLogging verwendet.

### **persistence/ModelPersistence.py**
- **ModelPersistence**: Stellt Methoden zum Speichern und Laden von PyTorch-Modellen bereit. Wird von ModelLogging verwendet.

### **CombinedClassifier.py**
- **CombinedClassifier**: Kombinierter Klassifikator, der zwei Module vereint: ein beliebiges Modell für feingranulare Klassifikation (TM1) und ein eigenes CNN-Modul für Oberklassen (TM2). Die finale Vorhersage kombiniert beide Ausgaben.
- Wird als Option im EMNISTModel verwendet.

### **CombinedClassifier2.py**
- **CombinedClassifier2**: Alternative Implementierung eines kombinierten Klassifikators, bei dem beide Module als MLP realisiert sind. Ebenfalls für Experimente mit alternativen Modellarchitekturen gedacht.

---

## Zusammenspiel der Klassen

1. **EMNISTModel** ist die zentrale Steuerungsklasse. Sie nutzt:
   - **PrepareData** zur Datenaufbereitung (Filtern, Rebalancieren, Aufteilen).
   - **DatasetEMNIST** für das finale Dataset-Objekt mit Augmentierung.
   - **ModelLogging** für Logging und Modell-Checkpoints.
   - **CombinedClassifier** oder **CombinedClassifier2** als Modellarchitektur (optional).
2. **ModelLogging** verwendet intern **LoggingTB** (TensorBoard-Logging) und **ModelPersistence** (Speichern/Laden von Modellen).
3. **HyperparameterTuning** nutzt **EMNISTModel** für die Durchführung der Trainingsläufe mit verschiedenen Parametern.
4. **OneOf** wird von **DatasetEMNIST** für die zufällige Auswahl von Augmentierungen verwendet.

---

## Beispielhafter Ablauf

1. **Datenaufbereitung:**  
   `PrepareData` filtert und balanciert die EMNIST-Daten, gibt Trainings-, Validierungs- und Testdaten zurück.

2. **Dataset-Erstellung:**  
   `DatasetEMNIST` erstellt daraus ein PyTorch-kompatibles Dataset mit Label-Mapping und Augmentierung.

3. **Modelltraining:**  
   `EMNISTModel.train()` trainiert das Modell (z.B. ResNet18 oder CombinedClassifier), nutzt dabei `ModelLogging` für Logging und Checkpoints.

4. **Modelltest:**  
   `EMNISTModel.test()` testet das Modell und gibt Metriken sowie die Konfusionsmatrix aus.

5. **Hyperparameter-Tuning:**  
   `HyperparameterTuning` automatisiert die Suche nach optimalen Trainingsparametern mit Optuna.

---

## Hinweise

- Die wichtigsten Einstiegspunkte für Experimente sind die Klassen **EMNISTModel** (Training/Test) und **HyperparameterTuning** (Optuna).
- Die Konfiguration der Modelle und Trainingsparameter erfolgt über Dictionaries, die an die jeweiligen Methoden übergeben werden.
- Für eigene Experimente können die Konfigurationsbeispiele aus den Konsolen-Outputs oder mainForShell.py übernommen werden.

---

