import os 
import sys
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelEvaluationArtifacts,ModelTrainerArtifacts,DataTransformationArtifacts
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object,save_object,load_numpy_arr_data
from sklearn.metrics import f1_score
class ModelEvaluation:
    def __init__(self,model_trainer_artifact:ModelTrainerArtifacts,model_evaluation_configs:ModelEvaluationConfig,data_tranformation_artifact:DataTransformationArtifacts):
        try:
            self.model_trainer_artifact=model_trainer_artifact
            self.model_evaluation_configs=model_evaluation_configs
            self.data_tranformation_artifact=data_tranformation_artifact

        except Exception as e :
            raise MyException(e,sys)

    def model_registory_on_aws(self)->ModelEvaluationArtifacts:
        try:
            raise Exception("aws code is not ready ")
        except Exception as e :
            raise MyException(e,sys)

    def model_registory_local(self)->ModelEvaluationArtifacts:
        try:
            prod_model_path=os.path.join(self.model_evaluation_configs.production_model_dir_path,self.model_evaluation_configs.s3_model_key_path)
            prod_model_exist=os.path.exists(prod_model_path)
            if prod_model_exist:
                #the prod model is already there so we have to made pred on the 
                # prod model on the test.npy . and check it's accuracy score with 
                # the model which we have just trained
                prod_model=load_object(prod_model_path)
                test_npy=load_numpy_arr_data(self.data_tranformation_artifact.transformed_test_file_path)
                test_x=test_npy[:,:-1]
                test_y=test_npy[:,-1]
                output=prod_model.predict(test_x)
                prod_f1_score=f1_score(test_y,output)
                logging.info(f"prod model f1 score {prod_f1_score}")
                # compare the f1 of prod+threshodl with trained model
                trained_model_performence_imporved:bool=prod_f1_score+self.model_evaluation_configs.change_threashold_score< self.model_trainer_artifact.metric_artifact.f1_score
                if trained_model_performence_imporved:
                    # newly trained model has pefomred better
                    logging.info("newly trained model has performed better then prod model ")
                    is_accepted=trained_model_performence_imporved
                    improved_acc=(self.model_trainer_artifact.metric_artifact.f1_score-prod_f1_score)
                else:
                    # newly traned model can't beat the acc of prod model
                    logging.info("newly trained model has not performed better then prod model ")
                    is_accepted=False
                    improved_acc=(self.model_trainer_artifact.metric_artifact.f1_score-prod_f1_score)
            else:
                # prod is empty
                logging.info("prod model dir is empty")
                is_accepted=True
                improved_acc=0

            model_evaluation_artifact=ModelEvaluationArtifacts(
                location_s3=self.model_evaluation_configs.location_s3,
                is_model_accepted=is_accepted,
                s3_model_path="",
                production_model_dir_path=self.model_evaluation_configs.production_model_dir_path,
                production_model_file_path=os.path.join(self.model_evaluation_configs.production_model_dir_path,self.model_evaluation_configs.s3_model_key_path),
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                improved_performance=improved_acc,
                data_tranformation_object_file_path=self.data_tranformation_artifact.transformed_object_file_path
                )
            return model_evaluation_artifact
        except Exception as e :
            raise MyException(e, sys)

    def initiate_model_evaluation(self)->ModelEvaluationArtifacts:
        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
            model_eval_artifact=self.model_registory_on_aws() if self.model_evaluation_configs.location_s3 else self.model_registory_local()
            logging.info(f"Exited initiate_model_evaluation method of ModelEvaluation class")
            logging.info(model_eval_artifact)
            return model_eval_artifact                
        except Exception as e:
            raise MyException(e,sys)