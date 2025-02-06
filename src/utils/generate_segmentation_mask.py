import os
import shutil
from collections import defaultdict

import copick
import copick_utils.writers.write as write
import numpy as np

# ### 2.Generate multi-class segmentation masks from picks, and saved them to the copick overlay directory
from copick_utils.segmentation import segmentation_from_picks
from tqdm import tqdm

from src.utils import load_settings

settings = load_settings()

config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 80,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 128],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 120,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 150,
            "map_threshold": 0.201
        }
    ],

    "overlay_root": "./input/working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "./input/czii-cryo-et-object-identification/train/static"
}"""

raw_data_dir = settings.raw_data_dir
train_data_working_dir = settings.train_data_working_dir
train_data_working_dir.mkdir(parents=True, exist_ok=True)
print(f"working dir: {train_data_working_dir}")

copick_config_path = str(train_data_working_dir / "copick.config")
output_overlay = str(train_data_working_dir / "overlay")

with open(copick_config_path, "w") as f:
    f.write(config_blob)

# Update the overlay
# Define source and destination directories
source_dir = str(raw_data_dir / "czii-cryo-et-object-identification/train/overlay")
destination_dir = str(train_data_working_dir / "overlay")

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    # Create corresponding subdirectories in the destination
    relative_path = os.path.relpath(root, source_dir)
    target_dir = os.path.join(destination_dir, relative_path)
    os.makedirs(target_dir, exist_ok=True)

    # Copy and rename each file
    for file in files:
        if file.startswith("curation_0_"):
            new_filename = file
        else:
            new_filename = f"curation_0_{file}"

        # Define full paths for the source and destination files
        source_file = os.path.join(root, file)
        destination_file = os.path.join(target_dir, new_filename)

        # Copy the file with the new name
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")


root = copick.from_file(copick_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"


# Just do this once
generate_masks = True

if generate_masks:
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]["label"] = object.label
            target_objects[object.name]["radius"] = object.radius

    for run in tqdm(root.runs):
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        target = np.zeros(tomo.shape, dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
            if len(pick):
                target = segmentation_from_picks.from_picks(
                    pick[0],
                    target,
                    target_objects[pickable_object.name]["radius"] * 0.4,
                    target_objects[pickable_object.name]["label"],
                )
        write.segmentation(run, target, copick_user_name, name=copick_segmentation_name)
