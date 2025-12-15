import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

print("Libraries imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Synthetic Dataset Generation for CNN Testing

def create_simple_image_dataset(n_samples=1000, image_size=(8, 8)):
    # Generate random images
    images = np.random.randn(n_samples, 1, image_size[0], image_size[1])

    center = image_size[0] //2
    labels = (images[:, 0, center, center] > 0).astype(int)

    return images, labels

# Create the dataset
X, y = create_simple_image_dataset(n_samples=1000, image_size=(8, 8))
print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Sample image shape: {X[0].shape}")

# Visualize a few samples
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    row = i // 4
    col = i % 4
    axes[row, col].imshow(X[i, 0], cmap='viridis')
    axes[row, col].set_title(f'Label: {y[i]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('few_samples.png', dpi=150, bbox_inches='tight')
print("Plot saved to few_samples.png")

def convolution_2d(input_matrix, kernel, stride=1, padding=0):
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant', constant_values=0)

    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Initialize output matrix
    output = np.zeros((output_h, output_w))

    # Perform convolution
    for i in range(0, output_h):
        for j in range(0, output_w):
            # Extract region of interest
            roi = input_matrix[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(roi * kernel)
    return output

# Test convolution operation
test_image = X[0, 0]
edge_kernel = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                        [-1, -1, -1]])

convolved = convolution_2d(test_image, edge_kernel, stride=1, padding=1)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_image, cmap='viridis')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(edge_kernel, cmap='RdBu')
axes[1].set_title('Edge Detection Kernel')
axes[1].axis('off')

axes[2].imshow(convolved, cmap='viridis')
axes[2].set_title('Convolved Result')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('convolution_2d.png', dpi=150, bbox_inches='tight')
print("Plot saved to convolution_2d.png")

print(f"Original shape: {test_image.shape}")
print(f"Kernel shape: {edge_kernel.shape}")
print(f"Output shape: {convolved.shape}")

# Max Pooling Operaiton from Scratch
def max_pooling_2d(input_matrix, pool_size=2, stride=2):
    
    input_h, input_w = input_matrix.shape

    # Calculate output dimensions
    output_h = (input_h - pool_size) // stride + 1
    output_w = (input_w - pool_size) // stride + 1

    # Initialize output matrix
    output = np.zeros((output_h, output_w))

    # Perform max pooling
    for i in range(output_h):
        for j in range(output_w):
            pool_region = input_matrix[i*stride:i*stride+pool_size,
                                       j*stride:j*stride+pool_size]
            output[i, j] = np.max(pool_region)

    return output

# Test pooling on convolved results
pooled = max_pooling_2d(convolved, pool_size=2, stride=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(convolved, cmap='viridis')
axes[0].set_title(f'Convolved Feature Map\nShape: {convolved.shape}')
axes[0].axis('off')

axes[1].imshow(pooled, cmap='viridis')
axes[1].set_title(f'Map Pooled Feature Map\nShape: {pooled.shape}')
axes[1].axis('off')

axes[2].plot(convolved.flatten(), label='Before Pooling', alpha=0.7)
axes[2].plot(pooled.flatten(), label='After Pooling', alpha=0.7)
axes[2].set_title('Feature Values Comparison')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('max_pooling_2d.png', dpi=150, bbox_inches='tight')
print("Plot saved to max_pooling_2d.png")

print(f"Original convolved shape: {convolved.shape}")
print(f"After max pooling shape: {pooled.shape}")
print(f"Dimensionality reduction: {convolved.size} -> {pooled.size} ({pooled.size/convolved.size*100:.1f}%)")



