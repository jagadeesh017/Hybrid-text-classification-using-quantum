from hqnlp import load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert config.model.model_type == "hybrid"
    assert config.data.dataset_name == "imdb"
