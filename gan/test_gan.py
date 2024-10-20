import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import *

# Test function for Discriminator
def test_discriminator():
    # Create an instance of the Discriminator
    discriminator = Discriminator()
    
    # Test with a batch of images of shape (batch_size, 3, H, W)
    batch_size = 16
    image_channels = 3
    image_size = 32  # Assuming the input image size is 32x32

    # Create a batch of random images (e.g., 16 RGB images of size 32x32)
    x = torch.randn(batch_size, image_channels, image_size, image_size)
    
    # Pass the images through the discriminator
    output = discriminator.forward(x)
    
    # The output should have shape (batch_size, 1) because the dense layer outputs a single value per image
    expected_shape = (batch_size, 1)
    
    # Check if the output shape is as expected
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    print(f"Discriminator test passed! Output shape: {output.shape}")

# Test function for Generator
def test_generator():
    # Create an instance of the Generator with a starting image size of 4
    generator = Generator(starting_image_size=4)
    
    # Test forward_given_samples method
    batch_size = 16
    latent_dim = 128
    z = torch.randn(batch_size, latent_dim)
    
    output = generator.forward_given_samples(z)
    
    # Expected output shape: (batch_size, 3, H, W), assuming the output image size is 32x32 after upsampling
    expected_shape = (batch_size, 3, 32, 32)  # Adjust the (32, 32) based on your model's output size
    assert output.shape == expected_shape, f"forward_given_samples output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    print(f"forward_given_samples passed! Output shape: {output.shape}")
    
    # Test forward method
    n_samples = 1024
    output = generator.forward(n_samples)
    
    # Expected output shape for the forward method
    expected_shape = (n_samples, 3, 32, 32)  # Adjust if the output image size is different
    assert output.shape == expected_shape, f"forward output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    print(f"forward passed! Output shape: {output.shape}")
    
    # Test dense layer reshaping
    out_dense = generator.dense(z)
    assert out_dense.shape == (batch_size, 2048), f"Dense layer output shape mismatch: expected {(batch_size, 2048)}, got {out_dense.shape}"
    
    out_reshaped = out_dense.view(batch_size, 128, 4, 4)  # starting_image_size = 4
    assert out_reshaped.shape == (batch_size, 128, 4, 4), f"Reshape output shape mismatch: expected {(batch_size, 128, 4, 4)}, got {out_reshaped.shape}"
    
    print(f"Dense layer reshape passed! Reshaped output: {out_reshaped.shape}")


# Test function for ResBlock
def test_resblock():
    # Parameters
    batch_size = 4
    input_channels = 128
    height, width = 64, 64  # Input spatial dimensions
    n_filters = 128

    # Create a random input tensor with shape (batch_size, input_channels, height, width)
    x = torch.randn(batch_size, input_channels, height, width)

    # Instantiate the ResBlockDown with input_channels and n_filters
    res_block = ResBlock(input_channels=input_channels, kernel_size=3, n_filters=n_filters)

    # Forward pass through ResBlockDown
    output = res_block(x)

    # Check the output shape: since downsampling is applied, the spatial dimensions should be halved
    expected_height = height
    expected_width = width
    expected_shape = (batch_size, n_filters, expected_height, expected_width)

    # Assert that the output shape matches the expected shape
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"

    print(f"Test passed! Output shape: {output.shape}")

# Test function for ResBlockDown
def test_resblockdown():
    # Parameters
    batch_size = 4
    input_channels = 3
    height, width = 64, 64  # Input spatial dimensions
    n_filters = 128

    # Create a random input tensor with shape (batch_size, input_channels, height, width)
    x = torch.randn(batch_size, input_channels, height, width)

    # Instantiate the ResBlockDown with input_channels and n_filters
    res_block_down = ResBlockDown(input_channels=input_channels, kernel_size=3, n_filters=n_filters)

    # Forward pass through ResBlockDown
    output = res_block_down(x)

    # Check the output shape: since downsampling is applied, the spatial dimensions should be halved
    expected_height = height // 2
    expected_width = width // 2
    expected_shape = (batch_size, n_filters, expected_height, expected_width)

    # Assert that the output shape matches the expected shape
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"

    print(f"Test passed! Output shape: {output.shape}")

# Test function for ResBlockUp
def test_resblockup():
    # Parameters
    batch_size = 4
    input_channels = 64
    height, width = 32, 32  # Input spatial dimensions
    n_filters = 128
    upscale_factor = 2

    # Create a random input tensor with shape (batch_size, input_channels, height, width)
    x = torch.randn(batch_size, input_channels, height, width)

    # Instantiate the ResBlockUp with input_channels and n_filters
    res_block_up = ResBlockUp(input_channels=input_channels, kernel_size=3, n_filters=n_filters)

    # Forward pass
    output = res_block_up(x)

    # Check the output shape
    expected_height = height * upscale_factor
    expected_width = width * upscale_factor
    expected_shape = (batch_size, n_filters, expected_height, expected_width)

    # Assert that the output shape matches the expected shape
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"

    print(f"Test passed! Output shape: {output.shape}")

def main():
    batch_size, in_channels, height, width = 1, 3, 64, 64
    upscale_factor = 2

    # Create a random tensor
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")

    # Create the upsampling model and run it
    print("Testing UpSampleConv2D")
    model = UpSampleConv2D(in_channels)
    output = model(x)
    print(f"UpSampleConv2D output shape: {output.shape}")
    print()

    # Create the downsampling model and run it
    print("Testing DownSampleConv2D")
    model = DownSampleConv2D(output.shape[1])
    output = model(output)
    print(f"DownSampleConv2D output shape: {output.shape}")
    print()

    # Test ResBlockUp
    print("Testing ResBlockUp")
    test_resblockup()
    print()

    # Test ResBlockDown
    print("Testing ResBlockDown")
    test_resblockdown()
    print()

    # Test ResBlock
    print("Testing ResBlock")
    test_resblock()
    print()

    # Test the Generator
    print("Testing Generator")
    test_generator()
    print()

    # Test the Discriminator
    print("Testing Discriminator")
    test_discriminator()
    print()


if __name__ == "__main__":
    main()