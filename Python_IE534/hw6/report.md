
## Test accuracy

I used 196 features and trained the model for 200 epochs on the dataset `CIFAR10` for the `discriminator-with-generator` case, while 100 epochs for `discriminator-without-generator` case. 

* without the generator: **0.872** for 100 epochs.
* with the generator is **0.839** for 200 epochs (historical best is **0.846** on the epoch 192).


## Generated images from the trained Models

The generated images for epoch 1, 50, 100, 150, 200 are given below. Visually, the outputs for classes like, Horse and Truck are relatively good, while they are awful for Cat and Deer.

* Epoch 1
    
    ![epoch-001](./output/000.png)
* Epoch 50
    
    ![epoch-050](./output/049.png)
* Epoch 100
    
    ![epoch-100](./output/098.png)
* Epoch 150

    ![epoch-150](./output/147.png)
* Epoch 200
    
    ![epoch-200](./output/199.png)

# Real vs Perturbed

Real imgaes, their gradients and perturbed images are given below.

* Real Images
    
    ![real](./visualization/real_images.png)
* Gradients
    
    ![grad](./visualization/gradient_image.png)
* Perturbed Images
    
    ![jittered](./visualization/jittered_images.png)


## Synthetic Images Maximizing Classification Output

* Synthetic images without generator
    
    ![](./visualization/max_class_without_generator.png)

* Synthetic images with generator
    
    ![](./visualization/max_class_with_generator.png)

## Synthetic images maximizing features at various layers

* Synthetic images without generator
    * Layer 2
        ![](./visualization/max_features_without_generator_layer_2.png)
    * Layer 4
        ![](./visualization/max_features_without_generator_layer_4.png)
    * Layer 8
        ![](./visualization/max_features_without_generator_layer_8.png)

* Synthetic images with generator
    
    * Layer 2
        ![](./visualization/max_features_with_generator_layer_2.png)
    * Layer 4
        ![](./visualization/max_features_with_generator_layer_4.png)
    * Layer 8
        ![](./visualization/max_features_with_generator_layer_8.png)


