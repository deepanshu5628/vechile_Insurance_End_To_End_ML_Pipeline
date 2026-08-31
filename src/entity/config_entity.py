import os
from dataclasses import dataclass
from src.constants import (PIPELINE_NAME ,TIMESTAMP,ARTIFACT_DIR,DATA_INGESTION_DIR_NAME,DATA_INGESTION_COLLECTION_NAME,
                           DATA_INGESTION_TEST_TRAIN_SPLIT_RATIO,DATA_INGESTION_FEATURE_STORE_DIR,DATA_INGESTION_INGESTED_DIR,
                           FILE_NAME,TRAIN_FILE_NAME,TEST_FILE_NAME,DATA_VALIDATION_DIR_NAME,DATA_VALIDATION_REPORT_FILE_NAME,
                           DATA_TRANSFORMATION_DIR_NAME,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                           PREPROCSSING_OBJECT_FILE_NAME,MODEL_TRAINER_DIR_NAME,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME,
                           MODEL_TRAINER_EXPECTED_SCORE,MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,MODEL_TRAINER_N_ESTIMATORS,
                           MODEL_TRAINER_MIN_SAMPLES_SPLIT,MODEL_TRAINER_MIN_SAMPLES_LEAF,
                           MIN_SAMPLES_SPLIT_MAX_DEPTH,MIN_SAMPLES_SPLIT_CRITERION,
                           MIN_SAMPLES_SPLIT_RANDOM_STATE,MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
                           MODEL_BUCKET_NAME,MODEL_PUSHER_S3_KEY,MODEL_FILE_NAME,
                           PRODUCTION_MODEL_DIR_PATH,LOCATION_S3,RANDOM_STATE)

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str=PIPELINE_NAME
    artifact_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)   #artifcat/timestamp/
    timestamp:str=TIMESTAMP


training_pipeline_config:TrainingPipelineConfig=TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME) #artifcat/timestamp/data_ingestion
    feature_store_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)  #artifcat/timestamp/data_ingestion/feature_store/filename
    training_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)#artifcat/timestamp/data_ingestion/ingested/train.csv
    testing_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ratio:float=DATA_INGESTION_TEST_TRAIN_SPLIT_RATIO
    collection_name:str=DATA_INGESTION_COLLECTION_NAME
    random_state:int=RANDOM_STATE

@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir ,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformation_train_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FILE_NAME.replace("csv","npy"))
    transformation_test_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace("csv","npy"))
    transformation_object_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)
    random_state:int=RANDOM_STATE

@dataclass
class ModelTrainerConfig:
    model_trainer_dir:str=os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_file_path:str=os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy:float=MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path:str=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    n_estimators:int=MODEL_TRAINER_N_ESTIMATORS
    min_sample_split:int=MODEL_TRAINER_MIN_SAMPLES_SPLIT
    min_sample_leaf:int=MODEL_TRAINER_MIN_SAMPLES_LEAF
    max_depth:int=MIN_SAMPLES_SPLIT_MAX_DEPTH
    criterion:str=MIN_SAMPLES_SPLIT_CRITERION
    random_state:int=MIN_SAMPLES_SPLIT_RANDOM_STATE

@dataclass
class ModelEvaluationConfig:
    change_threashold_score:float=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name:str=MODEL_BUCKET_NAME
    s3_model_key_path:str=MODEL_FILE_NAME
    location_s3:bool=LOCATION_S3
    production_model_dir_path:str=PRODUCTION_MODEL_DIR_PATH

@dataclass
class ModelPusherConfg:
    bucket_name:str=MODEL_BUCKET_NAME
    s3_model_key_path:str=MODEL_FILE_NAME

@dataclass
class PredictionPipelineConfig:
    preprocessor_path:str=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)
    model_path:str=os.path.join(PRODUCTION_MODEL_DIR_PATH,MODEL_FILE_NAME)