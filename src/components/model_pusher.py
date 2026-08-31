import os
import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfg
from src.entity.artifact_entity import ModelEvaluationArtifacts
from src.utils.main_utils import save_object,load_object
from src.entity.artifact_entity import ModelPusherArtifacts


class ModelPusher:
    def __init__(self,model_pusher_configs:ModelPusherConfg,model_evaluation_artifact:ModelEvaluationArtifacts):
        try:
            self.model_pusher_configs=model_pusher_configs
            self.model_evaluation_artifact=model_evaluation_artifact
        except Exception as e:
            raise MyException(e,sys)

    def save_to_local(self,newly_trained_model:object,transformer_model:object):
        try:
            logging.info("saving newly trained model to local ")
            # make the folder if it does'nt exist
            transformer_save_path=os.path.join(self.model_evaluation_artifact.production_model_dir_path,self.model_pusher_configs.s3_transformer_key_path)
            os.makedirs(self.model_evaluation_artifact.production_model_dir_path,exist_ok=True)
            save_object(self.model_evaluation_artifact.production_model_file_path,newly_trained_model)
            save_object(transformer_save_path,transformer_model)
            file_path=self.model_evaluation_artifact.production_model_file_path
            logging.info("successfully saved newly trained model to local ")
            return file_path
        except Exception as e:
            raise MyException(e,sys)
        
    def save_to_aws(self,new_trained_model:object,transformer_model:object):
        try:
            return "path/model.pkl"
        except Exception as e:
            raise MyException(e,sys)

    def initiate_model_pusher(self)->ModelPusherArtifacts:
        try:
            # load the newly trained model
            newly_trained_model=load_object(self.model_evaluation_artifact.trained_model_path)
            transformer_model=load_object(self.model_evaluation_artifact.data_tranformation_object_file_path)
            file_path=self.save_to_aws(newly_trained_model,transformer_model) if self.model_evaluation_artifact.location_s3 else self.save_to_local(newly_trained_model,transformer_model)
            return ModelPusherArtifacts(bucket_name=self.model_pusher_configs.bucket_name,
                                s3_model_path=file_path,
                                location_s3=self.model_evaluation_artifact.location_s3,
                                production_model_dir_path=self.model_evaluation_artifact.production_model_dir_path,
                                production_model_file_path=self.model_evaluation_artifact.production_model_file_path)
        except Exception as e :
            raise MyException(e,sys)