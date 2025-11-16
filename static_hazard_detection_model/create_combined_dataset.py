import os
import shutil
import random
import yaml
import json
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths to your THREE processed YOLO datasets
PHELE_PATH = Path(r"C:\Users\Akil\Documents\Code\watai\CV\static_detection_model\unprocessed_data\PHELE Completely Labelled Image Dataset for Physical Hazards of the Elderly Living Environment\processed_hazards_dataset_yolo")
MIT_PATH = Path(r"C:\Users\Akil\Documents\Code\watai\CV\static_detection_model\unprocessed_data\MIT Indoor Dataset\processed_mit_yolo")
OCID_PATH = Path(r"C:\Users\Akil\Documents\Code\watai\CV\static_detection_model\unprocessed_data\OCID_partial\coco_clutter_dataset\processed_ocid_filtered_yolo")


# Output path
OUTPUT_DIR = Path("./final_fall_hazard_dataset_chris_mit_data")


# Define your final target classes (This order must be 100% consistent)
TARGET_CLASSES = [
    'rug', 
    'clutter_zone', 
    'wire', 
    'uneven_threshold', 
    'furniture', 
    'staircase'
]

# (To fix class imbalance)
MAX_IMAGES_PER_CLASS = 350

# (To hit 40% negative target)
TARGET_NEGATIVE_RATIO = 0.40

# (To create final split)
TRAIN_RATIO = 0.70 # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def setup_directories():
    """Creates the final output directories."""
    print(f"Creating output directories in {OUTPUT_DIR}...")
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def discover_data(sources):
    """Scans all source datasets and catalogs positive and negative images."""
    all_positive_images = []
    all_negative_images = []
    
    print("Scanning source directories...")
    for source_name, source_path in sources.items():
        if not source_path.exists():
            print(f"Warning: Path not found {source_path}. Skipping.")
            continue
            
        for split in ["train", "val"]: # Check both train/val folders in source
            image_dir = source_path / "images" / split
            label_dir = source_path / "labels" / split
            
            if not image_dir.exists() or not label_dir.exists():
                print(f"Warning: {split} folder not found in {source_name}. Skipping.")
                continue

            for img_path in tqdm(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")), desc=f"Scanning {source_name}/{split}"):
                label_path = label_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            classes_in_image = set()
                            lines = f.read().strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    classes_in_image.add(class_id)
                            
                            if classes_in_image:
                                all_positive_images.append({
                                    "image_path": img_path,
                                    "label_path": label_path,
                                    "classes": list(classes_in_image),
                                    "source": source_name
                                })
                            else:
                                # Label file exists but is empty = negative
                                all_negative_images.append({
                                    "image_path": img_path,
                                    "source": source_name
                                })
                    except Exception as e:
                        print(f"Error reading {label_path}: {e}")
                else:
                    # No label file exists = negative
                    all_negative_images.append({
                        "image_path": img_path,
                        "source": source_name
                    })

    print(f"\nDiscovered {len(all_positive_images)} total positive images.")
    print(f"Discovered {len(all_negative_images)} total negative images.")
    return all_positive_images, all_negative_images

def get_class_image_map(positive_images):
    """Creates a dictionary mapping class_id to a list of images containing it."""
    class_image_map = defaultdict(list)
    for img_data in positive_images:
        for class_id in img_data["classes"]:
            class_image_map[class_id].append(img_data)
    return class_image_map

def balance_positives(class_image_map, all_positive_images):
    """Undersamples over-represented classes to balance the dataset."""
    print(f"\nBalancing positive images. Capping at {MAX_IMAGES_PER_CLASS} images per class...")
    
    balanced_image_paths = set() # Use a set to avoid duplicating images
    
    for class_id, class_name in enumerate(TARGET_CLASSES):
        images = class_image_map.get(class_id, [])
        print(f"  Class '{class_name}' (ID {class_id}): {len(images)} images found.")
        
        if len(images) > MAX_IMAGES_PER_CLASS:
            print(f"    -> Undersampling to {MAX_IMAGES_PER_CLASS} images.")
            images_to_add = random.sample(images, MAX_IMAGES_PER_CLASS)
        else:
            images_to_add = images
            
        for img_data in images_to_add:
            balanced_image_paths.add(img_data["image_path"])

    # Convert set of paths back to list of dicts
    path_to_data_map = {img["image_path"]: img for img in all_positive_images}
    balanced_positive_list = [path_to_data_map[path] for path in balanced_image_paths]
    
    print(f"\nTotal positive images after balancing: {len(balanced_positive_list)}")
    return balanced_positive_list

def sample_negatives(num_balanced_positives, all_negative_images):
    """Calculates and samples the required number of negative images."""
    n_positive = num_balanced_positives
    
    # Calculate how many negatives we need for the 40% ratio
    # N_positive = (1 - 0.40) * N_total = 0.6 * N_total
    # N_total = N_positive / 0.6
    # N_negative = N_total * 0.40
    # N_negative = (N_positive / 0.6) * 0.4 = N_positive * (2/3)
    n_negative_target = int(n_positive * (TARGET_NEGATIVE_RATIO / (1.0 - TARGET_NEGATIVE_RATIO)))
    
    print(f"\nTargeting {TARGET_NEGATIVE_RATIO*100}% negative data.")
    print(f"  {n_positive} positive images requires {n_negative_target} negative images.")

    if n_negative_target > len(all_negative_images):
        print(f"  Warning: Not enough negatives ({len(all_negative_images)}). Using all available.")
        n_negative_target = len(all_negative_images)
        
    sampled_negatives = random.sample(all_negative_images, n_negative_target)
    print(f"  Randomly sampled {len(sampled_negatives)} negative images.")
    return sampled_negatives

def create_final_dataset(positive_list, negative_list, class_image_map):
    """Creates the final stratified train/val/test splits and copies files."""
    print("\nCreating final train/val/test splits...")

    # Create stratification keys (rarest class in each image)
    stratify_keys = []
    for img_data in positive_list:
        # Find the rarest class in this image to use as its "group" for splitting
        rarest_class_id = min(img_data["classes"], key=lambda cid: len(class_image_map.get(cid, [])))
        stratify_keys.append(rarest_class_id)

    # 1. Split Positives (Train and Temp)
    pos_train, pos_temp, y_train, y_temp = train_test_split(
        positive_list, stratify_keys,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=stratify_keys
    )
    
    # 2. Split Temp (Val and Test)
    # The new test_size is relative to the *temp* set
    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    try:
        pos_val, pos_test = train_test_split(
            pos_temp,
            test_size=relative_test_ratio,
            random_state=RANDOM_SEED,
            stratify=y_temp # Stratify the second split as well
        )
    except ValueError:
        print("Warning: Could not stratify val/test split (likely too few samples). Doing random split.")
        pos_val, pos_test = train_test_split(pos_temp, test_size=relative_test_ratio, random_state=RANDOM_SEED)

    # 3. Split Negatives
    neg_train, neg_temp = train_test_split(negative_list, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED)
    neg_val, neg_test = train_test_split(neg_temp, test_size=relative_test_ratio, random_state=RANDOM_SEED)

    # 4. Combine and print final counts
    splits = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test
    }
    
    print("\nFinal Split Counts:")
    print(f"  Train: {len(pos_train):<4} positive / {len(neg_train):<4} negative = {len(splits['train']):<5} total")
    print(f"  Val:   {len(pos_val):<4} positive / {len(neg_val):<4} negative = {len(splits['val']):<5} total")
    print(f"  Test:  {len(pos_test):<4} positive / {len(neg_test):<4} negative = {len(splits['test']):<5} total")
    
    final_stats = defaultdict(lambda: defaultdict(int))

    # 5. Copy files to new destination
    for split_name, image_list in splits.items():
        print(f"\nCopying {split_name} files...")
        for img_data in tqdm(image_list):
            try:
                img_path = img_data["image_path"]
                # Create a unique new name: e.g., "mit_airport_inside_0001.jpg"
                new_stem = f"{img_data['source']}_{img_path.stem}"
                
                # Copy image
                dest_img = OUTPUT_DIR / "images" / split_name / f"{new_stem}{img_path.suffix}"
                shutil.copy(img_path, dest_img)
                
                if "label_path" in img_data:
                    label_path = img_data["label_path"]
                    dest_label = OUTPUT_DIR / "labels" / split_name / f"{new_stem}.txt"
                    shutil.copy(label_path, dest_label)
                    
                    # Record class stats
                    for class_id in img_data["classes"]:
                        final_stats[split_name][TARGET_CLASSES[class_id]] += 1
                else:
                    final_stats[split_name]["negative"] += 1
            except Exception as e:
                print(f"Error copying {img_path}: {e}")

    # 6. Create data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_data = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # Add the test set
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, indent=2, sort_keys=False)

    print(f"\nFinal dataset and {yaml_path.name} created successfully.")
    print("\n--- Final Class Distribution (by image) ---")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    
    if "your_path" in str(PHELE_PATH) or \
       "your_path" in str(MIT_PATH) or \
       "your_path" in str(OCID_PATH):
        
        print("="*50)
        print("ERROR: Please update the placeholder paths at the top of the script.")
        print(f"PHELE_PATH = {PHELE_PATH}")
        print(f"MIT_PATH = {MIT_PATH}")
        print(f"OCID_PATH = {OCID_PATH}")
        print("="*50)
    else:
        setup_directories()
        
        all_positive, all_negative = discover_data({
            "phele": PHELE_PATH,
            "mit": MIT_PATH,
            "ocid": OCID_PATH
        })
        
        if not all_positive:
            print("\nFATAL ERROR: No positive images were found. Check your dataset paths.")
        else:
            class_image_map = get_class_image_map(all_positive)
            
            balanced_positive = balance_positives(class_image_map, all_positive)
            
            sampled_negative = sample_negatives(len(balanced_positive), all_negative)
            
            create_final_dataset(balanced_positive, sampled_negative, class_image_map)