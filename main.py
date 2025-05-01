from loguru import logger
from src.dataset import process_data
from src.features import create_features
from src.plots import create_plots
from src.modeling.train import fetch_data,train,save_model
from src.modeling.predict import evaluate

model_type = ["lr","dt"]
def main():
    source_dir = r"C:\ITI\mlops\No_trained_pipeline\Assignment_1\data\processed"
    logger.info("Pipeline Started............")
    create_features("train.csv")
    create_plots("Feat_train.csv")
    process_data("Feat_train.csv","passengerid")
    x_train,y_train,x_test,y_test  = fetch_data(source_dir)
    for m in model_type:
        model = train(m, x_train, y_train)
        save_model(model, m)
        evaluate(x_test, y_test, m)
    logger.success("Pipeline finished sucessfully")

main()