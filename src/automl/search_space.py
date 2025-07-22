search_space = {
    "model": {
        "type": "categorical",
        "choices": ["resnet18", "mobilenet_v2", "efficientnet_b0"],
    },
    "lr": {
        "type": "float",
        "log": True,
        "lower": 1e-5,
        "upper": 1e-2,
    },
    "weight_decay": {
        "type": "float",
        "log": True,
        "lower": 1e-6,
        "upper": 1e-3,
    },
    "optimizer": {
        "type": "categorical",
        "choices": ["adam", "sgd"],
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128],
    },
}
