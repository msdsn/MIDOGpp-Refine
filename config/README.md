# MIDOG Dataset Configuration Files

This folder contains sample configuration files for MIDOG dataset preparation.

## Available Configurations

### 1. `patches_640_single_class.yaml`
- Creates **640x640 patches**
- **Single class** detection (all objects as class 0)
- **Test set included** (train/val/test)
- **Skips empty patches**

### 2. `patches_480_multi_class.yaml`
- Creates **480x480 patches**
- **Multi-class** detection
- **Train/val only** (no test set)
- **Skips empty patches**

### 3. `full_images_multi_class.yaml`
- Uses **full images** (no patching)
- **Multi-class** detection
- **Test set included** (train/val/test)

### 4. `patches_640_overlap_160.yaml`
- Creates **640x640 patches**
- With **160 pixel overlap**
- **Single class** detection
- **Test set included**
- **Lower overlap threshold** (0.2)

## Usage

```bash
# With default configuration
python prepare_midog_dataset.py

# With custom configuration
python prepare_midog_dataset.py --config config/patches_640_single_class.yaml
python prepare_midog_dataset.py -c config/patches_480_multi_class.yaml
```

## Configuration Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `USE_PATCHES` | Create patches? | `true`, `false` |
| `PATCH_SIZE` | Patch size | `480`, `640`, `800` |
| `OVERLAP` | Patch overlap in pixels | `0`, `160`, `320` |
| `SINGLE_CLASS` | Single class mode? | `true`, `false` |
| `INCLUDE_TEST_SET` | Include test set? | `true`, `false` |
| `TRAIN_RATIO` | Training ratio | `0.7`, `0.8` |
| `VAL_RATIO` | Validation ratio | `0.15`, `0.2` |
| `SKIP_EMPTY_PATCHES` | Skip empty patches | `true`, `false` |
| `OVERLAP_RATIO_THRESHOLD` | Annotation overlap threshold | `0.2`, `0.3`, `0.5` |

## Creating New Configurations

You can copy existing files and modify them or create a new YAML file from scratch.

```yaml
# Example new configuration
USE_PATCHES: true
PATCH_SIZE: 512
OVERLAP: 64
SINGLE_CLASS: false
INCLUDE_TEST_SET: true
TRAIN_RATIO: 0.8
VAL_RATIO: 0.1
SKIP_EMPTY_PATCHES: true
OVERLAP_RATIO_THRESHOLD: 0.4
MIDOG_JSON: "MIDOG++.json"
IMAGES_DIR: "datasets/midog_original/images"
``` 