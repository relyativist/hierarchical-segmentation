# Hierarchical Semantic Segmentation

Implementation of hierarchical semantic segmentation model trained on Pascal-Part dataset. The model performs segmentation with 3 hierarchical levels:

 Level 1 (Fine): Background/Foreground segmentation (2 classes)

Level 2 (Middle): Background/Upper Body/Lower Body regions (3 classes)

Level 3 (Coarse): Detailed body parts segmentation (7 classes)

## Model
The model uses encoder-decoder architecture (models/encskipdec.py) with:

* ResNet-50 backbone pretrained on ImageNet
* Skip connections between encoder and decoder
* Three decoder branches for hierarchical predictions
* Tested with different loss functions:

Weighted Cross-Entropy Loss (baseline)

Tree-Min Loss (improved results)
```
@article{li2022deep,
  title={Deep Hierarchical Semantic Segmentation},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Li, Jianwu and Yang, Yi},
  journal={arXiv preprint arXiv:2203.14335},
  year={2022}
}
```

## Results
Model trained for 50 epochs on NVIDIA GeForce RTX 4090.
Hold-out test set results:

| Loss Function  | Level 1 mIoU | Level 2 mIoU | Level 3 mIoU |
|---------------|--------------|--------------|--------------|
| Weighted CE   | 0.75        | 0.62        | 0.35        |
| Tree-Min Loss | 0.78        | 0.65        | 0.40        |

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
