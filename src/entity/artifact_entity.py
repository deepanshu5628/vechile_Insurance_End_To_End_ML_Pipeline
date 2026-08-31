from dataclasses import dataclass
@dataclass
class DataIngestionArtifacts:
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifacts:
    validation_status:bool
    message:str
    validation_report_file_path:str

@dataclass
class DataTransformationArtifacts:
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_object_file_path:str

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    accuracy_score:float

@dataclass 
class ModelTrainerArtifacts:
    trained_model_file_path:str
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted:bool
    improved_performance:float
    s3_model_path:str
    trained_model_path:str
    production_model_dir_path:str
    production_model_file_path:str
    location_s3:bool
    data_tranformation_object_file_path:str

@dataclass 
class ModelPusherArtifacts:
    bucket_name:str
    s3_model_path:str
    location_s3:bool
    production_model_dir_path:str
    production_model_file_path:str
