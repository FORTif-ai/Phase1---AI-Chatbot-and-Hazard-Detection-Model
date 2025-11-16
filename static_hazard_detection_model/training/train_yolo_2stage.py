import os
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# CONFIGURATION

DATA_YAML_PATH = SCRIPT_DIR / "final_fall_hazard_dataset" / "data.yaml"

# Root directory for all saved runs
PROJECT_SAVE_DIR = SCRIPT_DIR / "runs"

# Base model
MODEL_TO_USE = "yolov8s.pt" 

# Training parameters
STAGE_1_EPOCHS = 50   
STAGE_2_EPOCHS = 100  
IMAGE_SIZE = 640
PATIENCE_VALUE = 50   

def train_stage_1():
    """
    STAGE 1: Freeze the backbone.
    """
    print(f"\n--- Starting Stage 1: Training with frozen backbone ({STAGE_1_EPOCHS} epochs) ---")
    
    model = YOLO(MODEL_TO_USE)
    
    # Train the model with the backbone frozen
    results_stage1 = model.train(
        data=str(DATA_YAML_PATH),
        epochs=STAGE_1_EPOCHS,
        imgsz=IMAGE_SIZE,
        freeze=10,  
        project=str(PROJECT_SAVE_DIR / "detect"),
        name="stage1_freeze_backbone",
        device=0,
        patience=PATIENCE_VALUE,
        plots=True,
        cache='disk',
    )
    
    return model.trainer.save_dir


def train_stage_2(stage1_model_path):
    """
    STAGE 2: Fine-tune the entire network.
    """
    print(f"\n--- Starting Stage 2: Fine-tuning all layers ({STAGE_2_EPOCHS} epochs) ---")
    
    model = YOLO(stage1_model_path)
    
    results_stage2 = model.train(
        data=str(DATA_YAML_PATH),
        epochs=STAGE_2_EPOCHS,
        imgsz=IMAGE_SIZE,
        freeze=0,
        optimizer='AdamW', 
        lr0=0.001,  
        batch=8,    
        project=str(PROJECT_SAVE_DIR / "detect"),
        name="stage2_finetune_all",
        device=0,
        patience=PATIENCE_VALUE,
        plots=True,
        cache='disk',
    )
    
    return model.trainer.save_dir


if __name__ == "__main__":
    
    if "your_path" in str(DATA_YAML_PATH):
        print("="*50)
        print("ERROR: Please update the 'DATA_YAML_PATH' variable at the top of the script")
        print("       to point to the 'data.yaml' file in your 'final_fall_hazard_dataset' folder.")
        print("="*50)
    else:
        # Run the training
        print("Starting two-stage training process...")
        
        # Run Stage 1
        best_stage1_model_dir = train_stage_1()
        
        # The 'best.pt' from stage 1 is located in its results directory
        stage1_weights_path = Path(best_stage1_model_dir) / "weights" / "best.pt"
        
        # Run Stage 2
        best_stage2_model_dir = train_stage_2(stage1_weights_path)
        
        print("\n--- Training Complete! ---")
        print(f"Stage 1 results saved in: {best_stage1_model_dir}")
        print(f"Final model results saved in: {best_stage2_model_dir}")
        print(f"\nYour best model is located at:")
        print(Path(best_stage2_model_dir) / "weights" / "best.pt")