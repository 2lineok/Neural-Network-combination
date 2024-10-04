
# Neural-Network
Recent advancements in neural networks have led to the development of various innovative architectures designed to approximate complex operators. These structures are increasingly being explored for their potential to enhance the performance and efficiency of machine learning models across a wide range of applications.
# Deep O Net
One of the novel architectures emerging from this trend is the Deep Operator Network, commonly known as Deep O Net. This method is designed to approximate complex operators by learning mappings between infinite-dimensional function spaces rather than just finite-dimensional vectors. 

Unlike traditional neural networks that focus on approximating functions $u(x)$, Deep O Net focuses on directly learning the operators $G$ that govern these functions. This makes it particularly powerful in applications involving partial differential equations (PDEs) or other scenarios where the underlying dynamics are best described by operators. Deep O Net structures its architecture to capture the intricate dependencies between inputs and outputs in these infinite-dimensional spaces.

$$
G(x)(y) \approx \sum_{i=1}^N \left[ \sum_{j=1}^M \sigma \left( \sum_{k=1}^s w_{ijk}u(x_k) + \theta_{ij} \right) \right] \sigma \left( \sum_{l=1}^t w_{il}y_l + \theta_i \right)
$$


# YOLO

YOLO(You Only Look Once) is a deep learning-based object detection model that can identify and locate objects within an image in a single forward pass through the network. Unlike traditional object detection methods that use a two-step approach (first generating region proposals and then classifying them), YOLO frames object detection as a single regression problem. This allows it to predict bounding boxes and class probabilities directly from full images with a single neural network evaluation, making it much faster than previous approaches.

If we have input as image the output is both localization and classification of multiple objects within an image. YOLO achieves this by dividing the image into a grid and predicting bounding boxes and class labels for each grid cell. If multiple objects are present, YOLO predicts multiple bounding boxes and classifies each one.


<div align="center">
  <img width="40%" src="https://raw.githubusercontent.com/2lineok/Neural-Network-combination/main/YOLO/Cup-bowl-detection-1/data/3221.jpg" style="display: inline-block; margin-right: 10px;">
  <img width="40%" src="https://raw.githubusercontent.com/2lineok/Neural-Network-combination/main/YOLO/runs/detect/predict2/3221.jpg" style="display: inline-block;">
</div>


# Combination of two methods

Consider an image, which is essentially a continuous function representing pixel intensities over a two-dimensional grid. For computational purposes, this continuous image is discretized into a grid of pixels. This grid represents the image in a form that a neural network can process and perform object detection.

<div align="center">
  <img width="40%" src="https://raw.githubusercontent.com/2lineok/Neural-Network-combination/main/Explanation/goat_image.jpg" style="display: inline-block; margin-right: 10px;">
  <img width="40%" src="https://raw.githubusercontent.com/2lineok/Neural-Network-combination/main/Explanation/goat_image_with_grid.jpg" style="display: inline-block;">
</div>

We can think of the image as a function $u(x)$, where $x$ represents the coordinates in the image grid. The operator $G$ takes the the discretized image $u$ as input and produces a result at a point $y$. In YOLO it would be width, height, cofidence and class probability because coordinates are already included. Mathematically, we can express this as $G(u)(y)$ and try to approximate the operator.
