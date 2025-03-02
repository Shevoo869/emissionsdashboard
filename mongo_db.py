import os
import json
import datetime
import logging
import shutil
from typing import Dict, Any, List, Tuple, Union
from pymongo import MongoClient


class ResultStatus:
    """Result Status Tracking"""
    def __init__(self):
        self.success = True
        self.errors = []
        self.warnings = []
        self.saved_files = []
        self.start_time = datetime.datetime.now()

    def add_error(self, error_msg: str):
        self.success = False
        self.errors.append(error_msg)

    def add_warning(self, warning_msg: str):
        self.warnings.append(warning_msg)

    def add_saved_file(self, file_path: str):
        self.saved_files.append(file_path)

    def get_report(self) -> Dict:
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        return {
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "saved_files": self.saved_files,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "total_files_saved": len(self.saved_files)
        }


class DLResultsManager:
    def __init__(self, db_url: str, db_name: str):
        """Initialize Manager"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Setup Logging
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"results_manager_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Connect to MongoDB
        try:
            self.client = MongoClient(db_url)
            self.db = self.client[db_name]
            self.client.server_info()  # Test connection
            self.logger.info("MongoDB connection successful")
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    def save_training_results(self,
                              experiment_name: str,
                              metrics: Dict[str, Any],
                              image_files: List[str],
                              csv_files: List[str],
                              model_file: str = None,
                              additional_info: Dict[str, Any] = None) -> Tuple[str, Dict]:
        status = ResultStatus()
        self.logger.info(f"Saving results for experiment '{experiment_name}'")
        
        try:
            # Create experiment folder
            folder_path = self._create_experiment_folder(experiment_name)
            self.logger.info(f"Experiment folder created: {folder_path}")
            
            # Save image files
            saved_images = {}
            for img_path in image_files:
                saved_path = self._save_file(img_path, os.path.join(folder_path, "images"), status)
                if saved_path:
                    saved_images[os.path.basename(img_path)] = saved_path
            
            # Save CSV files
            saved_csvs = {}
            for csv_path in csv_files:
                saved_path = self._save_file(csv_path, os.path.join(folder_path, "csv"), status)
                if saved_path:
                    saved_csvs[os.path.basename(csv_path)] = saved_path
            
            # Save model file
            model_path = None
            if model_file:
                model_path = self._save_file(model_file, os.path.join(folder_path, "models"), status)
            
            # Create MongoDB document
            result_doc = {
                'experiment_name': experiment_name,
                'timestamp': datetime.datetime.now(),
                'metrics': metrics,
                'image_files': saved_images,
                'csv_files': saved_csvs,
                'folder_path': folder_path,
                'status': status.get_report()
            }
            if model_path:
                result_doc['model_path'] = model_path
            if additional_info:
                result_doc['additional_info'] = additional_info

            # Save to MongoDB
            result = self.db.training_results.insert_one(result_doc)
            result_id = str(result.inserted_id)

            self.logger.info(f"Results saved, MongoDB ID: {result_id}")

            # Generate final report
            final_report = {
                'result_id': result_id,
                'experiment_name': experiment_name,
                'folder_path': folder_path,
                **status.get_report()
            }

            # Save report as JSON
            report_path = os.path.join(folder_path, 'save_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)

            return result_id, final_report
        except Exception as e:
            error_msg = f"Error during save process: {str(e)}"
            status.add_error(error_msg)
            self.logger.error(error_msg)
            return None, status.get_report()

    def _create_experiment_folder(self, experiment_name: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{experiment_name}_{timestamp}"
        base_path = os.path.join(self.base_dir, "results", folder_name)
        subdirs = ["images", "csv", "models"]
        for subdir in subdirs:
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
        return base_path

    def _save_file(self, source_path: str, target_dir: str, status: ResultStatus) -> Union[str, None]:
        try:
            if not os.path.exists(source_path):
                status.add_warning(f"File not found: {source_path}")
                self.logger.warning(f"File not found: {source_path}")
                return None
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_dir, filename)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(source_path, target_path)
            status.add_saved_file(target_path)
            self.logger.info(f"File saved successfully: {target_path}")
            return target_path
        except Exception as e:
            raise Exception(f"Failed to save file {source_path}: {str(e)}")


if __name__ == "__main__":
    manager = DLResultsManager(
        db_url="mongodb+srv://Patrick:11890214@cluster1.88l7u.mongodb.net/",
        db_name="deep_learning_results"
    )

    # Absolute paths for test files
    image_files = [
        "F:/MCU/Final Project/test061224/loss_plot.png",
        "F:/MCU/Final Project/test061224/accuracy_plot.png"
    ]
    csv_files = [
        "F:/MCU/Final Project/test061224/training_history.csv",
        "F:/MCU/Final Project/test061224/evaluation_results.csv"
    ]
    model_file = "F:/MCU/Final Project/test061224/model.h5"

    # Check file existence
    files = image_files + csv_files + [model_file]
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")

    # Save results
    result_id, report = manager.save_training_results(
        experiment_name="mnist_test",
        metrics={
            "final_loss": 0.234,
            "final_accuracy": 0.956
        },
        image_files=image_files,
        csv_files=csv_files,
        model_file=model_file,
        additional_info={"epochs": 100, "batch_size": 32}
    )

    # Print report
    print("\nFinal Report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))