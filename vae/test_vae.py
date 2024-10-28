from model import *
import torch.nn.functional as F

def test_vaeencoder():
    z = torch.randn(16, 3, 32, 32)
    encoder = VAEEncoder((16, 3, 32, 32), 256)
    mu, logvar = encoder(z)

    assert mu.shape == (16, 256) and logvar.shape == (16, 256)

    print(f"mu shape: {mu.shape}")
    print(f"logvar shape: {logvar.shape}")

def test_mse_loss():
    z = torch.randn(16, 3, 32, 32)
    encoder = Encoder((16, 3, 32, 32), 256)
    decoder = Decoder(256, (16, 3, 32, 32))
    out = decoder(encoder(z))    
    print(F.mse_loss(z, out))

def test_decoder():
    z = torch.randn(16, 256)
    model = Decoder(256, (16, 3, 32, 32))
    out = model(z)
    assert out.shape == (16, 3, 32, 32)

    print(f"Output shape: {out.shape}")

def test_encoder():
    z = torch.randn(16, 3, 32, 32)
    model = Encoder((16, 3, 32, 32), 256)
    out = model(z)
    assert out.shape == (16, 256)

    print(f"Output shape: {out.shape}")

def main():
    # Test the Encoder
    print("Testing Encoder")
    test_encoder()
    print()

    # Test the Decoder
    print("Testing Decoder")
    test_decoder()
    print()

    # Test MSE Loss
    print("Testing MSE Loss")
    test_mse_loss()
    print()

    # Test the VAEEncoder
    print("Testing VAE Encoder")
    test_vaeencoder()
    print()

if __name__ == "__main__":
    main()