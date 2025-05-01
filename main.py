from loguru import logger
from src.dataset import process_data
from src.features import create_features
from src.plots import create_plots
from src.modeling.train import fetch_data,train,save_model
from src.modeling.predict import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    source_dir = cfg.pipeline.model.processed_data_path
    logger.info("Pipeline Started............")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    create_features(cfg.pipeline.data)
    create_plots(cfg.pipeline)
    process_data(cfg.pipeline.data)
    x_train,y_train,x_test,y_test  = fetch_data(cfg.pipeline)
    model = train(cfg.pipeline.model, x_train, y_train)
    save_model(model, cfg.pipeline.model)
    evaluate(x_test, y_test, cfg.pipeline.evaluate)
    logger.success("Pipeline finished sucessfully")

main()