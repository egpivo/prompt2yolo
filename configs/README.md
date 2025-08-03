## Configuratoin
This section details the required and optional configuration files used in the YOLO training pipeline. These YAML files control aspects such as object detection classes, prompts for image generation, and YOLO model parameters.

### Required Configuration

- [input.yaml](configs/input.yaml): Contains the prompts and object classes.
     ```yaml
     classes:
       - person
       - car
       - bear

     prompts:
       - "Realtek style photo, camping site, dusk, person, fire, rain, 1 meter view"
       - "Realtek style photo, camping site, night, person, campfire, fog, 2 meters view"
     ```
     - `classes`: List of object classes you want the model to detect.
     - `prompts`: Descriptive prompts used for image generation.

### Optional Configurations
#### YOLO v5

##### Image Generation
- [image_generator.yaml](configs/yolo_v5/image_generator.yaml): Configures the DreamBooth-based image generation settings.
  ```yaml
  model_path: "stabilityai/stable-diffusion-xl-base-1.0"  # Path to the model
  vae_path: "madebyollin/sdxl-vae-fp16-fix"              # Path to the VAE model
  generator: "realtek"                                   # LoRA checkpoints. Options: 'pixart' or 'realtek'
  image_size:                                            # Dimensions of generated images (width, height)
    - 1024
    - 1024
  guidance_scale: 7                                      # Guidance scale for image generation
  steps: 40                                              # Number of inference steps
  num_images: 10                                         # Number of images to generate
  val_ratio: 0.1                                         # Proportion of validation dataset
  test_ratio: 0.2                                        # Proportion of test dataset
  negative_prompt:                                       # Avoid generation of unwanted elements
    "anime, cartoon, graphic, text, painting, crayon, graphite, abstract"
  ```
- [image_labeler.yaml](configs/yolo_v5/image_labeler.yaml): Defines the configuration for the YOLO-World model used for image labeling.
   ```yaml
   yolo_model: "yolov8l-world.pt"                        # Path to the YOLO model
   conf: 0.3                                             # Confidence threshold for predictions
   iou: 0.5                                              # IoU threshold for non-max suppression
   max_det: 15                                           # Maximum number of detections
   image_size: 640                                       # Size of the input images
   augment: true                                         # Whether to apply augmentations during prediction
   agnostic_nms: true                                    # Whether to use class-agnostic NMS
   ```

##### Model Training

- [training.yaml](configs/yolo_v5/training.yaml): Defines parameters for testing the trained YOLOv5 model.
     ```yaml
    epochs: 100                                            # Number of training epochs
    batch_size: 64                                         # Batch size for training
    optimizer: "Adam"                                      # Optimizer to use during training
    patience: 100                                          # Early stopping patience
     ```
##### Model Evaluation

- [evaluation.yaml](configs/yolo_v5/evaluation.yaml): YOLOv5 testing parameters.
     ```yaml
    confidence: 0.05                                       # Confidence threshold for inference
    iou: 0.4                                               # IoU threshold for non-max suppression during inference
    img_size: 512                                          # Image size for testing input
     ```

##### Inference: Label Detection

- [evaluation.yaml](configs/yolo_v5/evaluation.yaml): Configures parameters for label detection and evaluation tasks.

```yaml
data_config: "test_data.yaml"                          # Path to the test data configuration file
data_path: "test/images"                               # Path to the test image dataset
confidence: 0.5                                        # Confidence threshold for detection
iou: 0.45                                              # IoU threshold for non-max suppression
img_size: 1024                                         # Size of the input images
batch_size: 64                                         # Batch size for detection
directory: "model"                                     # Output directory for detection results
```

#### YOLO v3 Tiny


##### Image Generation
- [image_generator.yaml](configs/yolo_v5/image_generator.yaml): Configures the DreamBooth-based image generation settings.
  ```yaml
  model_path: "stabilityai/stable-diffusion-xl-base-1.0"  # Path to the model
  vae_path: "madebyollin/sdxl-vae-fp16-fix"              # Path to the VAE model
  generator: "realtek"                                   # LoRA checkpoints. Options: 'pixart' or 'realtek'
  image_size:                                            # Dimensions of generated images (width, height)
    - 1024
    - 1024
  guidance_scale: 7                                      # Guidance scale for image generation
  steps: 40                                              # Number of inference steps
  num_images: 10                                         # Number of images to generate
  negative_prompt:                                       # Avoid generation of unwanted elements
    "anime, cartoon, graphic, text, painting, crayon, graphite, abstract"
  ```
- [image_labeler.yaml](configs/yolo_v5/image_labeler.yaml): Defines the configuration for the YOLO-World model used for image labeling.
   ```yaml
   yolo_model: "yolov8l-world.pt"                        # Path to the YOLO model
   conf: 0.3                                             # Confidence threshold for predictions
   iou: 0.5                                              # IoU threshold for non-max suppression
   max_det: 15                                           # Maximum number of detections
   image_size: 640                                       # Size of the input images
   augment: true                                         # Whether to apply augmentations during prediction
   agnostic_nms: true                                    # Whether to use class-agnostic NMS
   ```

##### Model Training

- [training.yaml](configs/yolo_v3_tiny/training.yaml): Defines parameters for testing the trained YOLOv5 model.
    ```yaml
    batch_size: 64                                          # Batch size for training
    max_batches: 5000                                       # Maximum number of training iterations
    learning_rate: 0.001                                    # Initial learning rate for training
    steps:                                                  # Steps at which to reduce the learning rate
      - 4000
      - 4500
    momentum: 0.9                                           # Momentum value for optimization
    weight_decay: 0.0005                                    # Weight decay for regularization
    ```
