```mermaid
graph TD;
    A[Input Layer] --> B[Conv2d: 1x6x5x5]
    B --> C[ReLU]
    C --> D[MaxPool2d: 2x2]
    D --> E[Conv2d: 6x16x5x5]
    E --> F[ReLU]
    F --> G[MaxPool2d: 2x2]
    G --> H[Flatten: 16x5x5]
    H --> I[Linear: 400x32]
    I --> J[ReLU]
    J --> K[Linear: 32x16]
    K --> L[ReLU]
    L --> M[Linear: 16x2]
    M --> N[Output Layer]
```

### Detailed Explanation of Spatial Dimensions

1. **Input Layer**: The network starts with a 32x32 input image.

2. **First Convolutional Layer (Conv2d: 1x6x5x5)**:
   - The input image is convolved with 6 filters of size 5x5.
   - This reduces the spatial dimensions from 32x32 to 28x28 (since a 5x5 kernel reduces each dimension by 4).

3. **ReLU Activation**: Applies a ReLU activation function, which does not change the dimensions.

4. **First Max Pooling Layer (MaxPool2d: 2x2)**:
   - A 2x2 pooling operation with stride 2 is applied.
   - This reduces the dimensions from 28x28 to 14x14.

5. **Second Convolutional Layer (Conv2d: 6x16x5x5)**:
   - The 14x14x6 feature maps are convolved with 16 filters of size 5x5.
   - This reduces the spatial dimensions from 14x14 to 10x10.

6. **ReLU Activation**: Again, applies a ReLU activation function, which does not change the dimensions.

7. **Second Max Pooling Layer (MaxPool2d: 2x2)**:
   - Another 2x2 pooling operation with stride 2 is applied.
   - This reduces the dimensions from 10x10 to 5x5.

8. **Flatten Layer (Flatten: 16x5x5)**:
   - The 5x5x16 feature maps are flattened into a 1D vector of size 400 (16 channels * 5 height * 5 width).

9. **First Fully Connected Layer (Linear: 400x32)**:
   - The flattened vector is fed into a fully connected layer with 32 outputs.

10. **ReLU Activation**: Applies a ReLU activation function.

11. **Second Fully Connected Layer (Linear: 32x16)**:
   - The 32-dimensional vector is fed into another fully connected layer with 16 outputs.

12. **ReLU Activation**: Applies a ReLU activation function.

13. **Output Layer (Linear: 16x2)**:
   - The 16-dimensional vector is fed into the final output layer, producing 2 outputs (one for each class).

