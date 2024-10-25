# Hierarchical Semantic Segmentation

Implementation of hierarchical semantic segmentation model trained on Pascal-Part dataset. The model performs segmentation with 3 hierarchical levels.

Results of Encoder Decoder segmentatation models/encskipdec.py are in notebooks/training.ipynb, model was trained for 50 epochs on  NVIDIA GeForce RTX 4090

## Requirements

- Docker
- NVIDIA GPU with CUDA support
- Pascal-Part dataset with the following structure:
```
.
|-- JPEGImages
|-- classes.txt
|-- gt_masks
|-- train_id.txt
--- val_id.txt
```
## Installation

1. Clone this repository
2. Place Pascal-Part dataset in your local directory
3. Build Docker image:
```bash
docker build -t hieraseg docker/
```

## Usage

### Training

To train the model with default parameters:

```bash
docker run --rm -it --name pascal --gpus all \
  -v /path/to/pascal_data:/root/data \
  hieraseg python train.py -c config.yaml
```

### Evaluation
NOTE! You have to set to `True` make_logs and save_model in configs/config.yaml
To evaluate trained model from experiment from config `experiment_name` and visualize results:

```bash
docker exec -it pascal python eval.py \
  --experiment_name experiment_name \
  --visualize \
  --num_viz 10 \
  --viz_dir ./logs/experiment_name
```

### Monitoring Training

View training progress using TensorBoard:

```bash 
docker run --rm -it \
  -v /path/to/code:/workspace/logs:ro \
  -p 8080:8080 \
  tensorflow/tensorflow:latest \
  tensorboard --logdir /workspace/logs --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 in your browser.
