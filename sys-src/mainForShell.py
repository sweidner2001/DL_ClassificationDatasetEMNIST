import argparse
from main import HyperparameterTuning, EMNISTModel


def get_args():
    parser = argparse.ArgumentParser(
        description="Starte Training oder Hyperparameter-Tuning"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Subcommand: tuning
    tune_parser = subparsers.add_parser("HTuning", help="Hyperparameter-Tuning starten")
    tune_parser.add_argument("--Trials", type=int, default=10, help="Anzahl der Optuna-Trials")
    tune_parser.add_argument("--ExperimentName", type=str, default="HyperparameterTuning_1", help="ExperimentName")

    train_parser = subparsers.add_parser("Training", help="Training mit festen Parametern")
    train_parser.add_argument("--LoadBestTrail", type=bool, default=False, help="Soll der Beste Trail der Hyperparameter Optimierung geladen werden?")
    train_parser.add_argument("--HTuningExperimentName", type=str, default='', help="Experimentname der Hyperparameter-Optimierung")
    train_parser.add_argument("--Logging", type=bool, default=False, help="Logging")
    train_parser.add_argument("--ExperimentName", type=str, default="TrainingRun_1", help="ExperimentName")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()

    if args.mode == "HTuning":
        HyperparameterTuning(experiment_name=args.ExperimentName).startTuning(n_trails=args.Trials)
    elif args.mode == "Training":
        if args.LoadBestTrail:
            best_params = HyperparameterTuning().get_best_params(experiment_name=args.HTuningExperimentName)
            config = best_params
        else:

            # config_CombinedClassifier = {
            #     'linear_out_1': 256,
            #     'linear_out_2': 256,
            #     'dropout': 0.18,
            # }

            # config_CombinedClassifier = {
            #     'conv2d_out_1': 64,
            #     'conv2d_kernel_size_1': 5,
            #     'conv2d_out_2': 64,
            #     'conv2d_kernel_size_2': 3,
            #     'conv2d_out_3': 32,
            #     'conv2d_kernel_size_3': 3,
            #     'linear_out_1': 128,
            #     'linear_out_2': 32,
            #     'dropout': 0.32,
            # }
            config = {
                "lr": 0.000203,
                "weight_decay": 0.000205,
                "num_epochs": 13,
                "batch_size": 64,
                "only_train_classifier": False,
                "sched_patience": 5,
                "sched_factor": 0.442897,
                "sched_min_lr": 0.000001,
                'train_with_CombinedClassifier': False,
                'probabilities': [0.16, 0.51, 0.64],
                'config_CombinedClassifier': None
            }

            emnist = EMNISTModel(experiment_name=args.ExperimentName, validation_dataset_for_training=True)
            emnist.train(config=config, logging=True)
            emnist.test(test_with_CombinedClassifier=config['train_with_CombinedClassifier'])





