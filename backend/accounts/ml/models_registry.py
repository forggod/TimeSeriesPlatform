from .models import LSTMModel, GRUModel


MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
}

MODELS = {
    "lstm": {
        "name": "LSTM",
        "params": {
            "sequence_length": {
                "label": "Длина последовательности",
                "type": "number",
                "default": 20,
            },
            "hidden_size": {
                "label": "Размер скрытого слоя",
                "type": "number",
                "default": 64,
            },
            "num_layers": {
                "label": "Кол-во слоев",
                "type": "number",
                "default": 2,
            },
            "learning_rate": {
                "label": "Скорость обучения",
                "type": "number",
                "default": 0.001,
            },
            "epochs": {
                "label": "Кол-во эпох",
                "type": "number",
                "default": 20,
            }
        }
    },
    "gru": {
        "name": "GRU",
        "params": {
            "sequence_length": {
                "label": "Длина последовательности",
                "type": "number",
                "default": 20,
            },
            "hidden_size": {
                "label": "Размер скрытого слоя",
                "type": "number",
                "default": 64,
            },
            "num_layers": {
                "label": "Кол-во слоев",
                "type": "number",
                "default": 2,
            },
            "learning_rate": {
                "label": "Скорость обучения",
                "type": "number",
                "default": 0.001,
            },
            "epochs": {
                "label": "Кол-во эпох",
                "type": "number",
                "default": 20,
            }
        }
    }
}
