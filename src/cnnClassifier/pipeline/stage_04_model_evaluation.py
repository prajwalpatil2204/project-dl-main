import os
import sys
sys.path.append(os.getcwd())
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_evaluation import Evaluation
from src.cnnClassifier import logger
import os
import mlflow


STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        
        mlflow.set_tracking_uri("https://dagshub.com/prajwalpatil2204/project-dl-main.mlflow")

        os.environ["MLFLOW_TRACKING_USERNAME"]="prajwalpatil2204"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="93c20ae46965bedc7ee24237d553b393d8dc7523"

        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e