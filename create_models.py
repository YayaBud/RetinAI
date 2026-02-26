"""
Script to create and inspect the three specialized EfficientNet-B3 models.
This script creates the models without training them.
"""

import torch
from src.models import create_cnn1_model, create_cnn2_model, create_cnn3_model


def print_model_info(model, model_name, num_classes):
    """Print information about a model."""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Number of classes: {num_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: {output.shape[1]}")
    
    # Show model architecture summary
    print(f"\nModel Architecture:")
    print(f"  Backbone: EfficientNet-B3 (pretrained on ImageNet)")
    print(f"  Classification Head:")
    print(f"    - Global Average Pooling")
    print(f"    - Batch Normalization")
    print(f"    - Dropout (p=0.4)")
    print(f"    - Linear Layer ({output.shape[1]} classes)")
    
    return model


def main():
    """Create and display all three models."""
    print("\n" + "="*60)
    print("Creating Specialized EfficientNet-B3 Models")
    print("="*60)
    
    # CNN-1: Diabetic Retinopathy (5-class)
    print("\n[1/3] Creating CNN-1 for Diabetic Retinopathy Detection...")
    cnn1 = create_cnn1_model(pretrained=True, dropout=0.4)
    print_model_info(cnn1, "CNN-1: Diabetic Retinopathy Detection", 5)
    print("\nClasses:")
    print("  0: No DR")
    print("  1: Mild")
    print("  2: Moderate")
    print("  3: Severe")
    print("  4: Proliferative DR")
    
    # CNN-2: Glaucoma (binary)
    print("\n[2/3] Creating CNN-2 for Glaucoma Detection...")
    cnn2 = create_cnn2_model(pretrained=True, dropout=0.4)
    print_model_info(cnn2, "CNN-2: Glaucoma Detection", 2)
    print("\nClasses:")
    print("  0: Normal")
    print("  1: Glaucoma")
    
    # CNN-3: Pathologic Myopia (binary)
    print("\n[3/3] Creating CNN-3 for Pathologic Myopia Detection...")
    cnn3 = create_cnn3_model(pretrained=True, dropout=0.4)
    print_model_info(cnn3, "CNN-3: Pathologic Myopia Detection", 2)
    print("\nClasses:")
    print("  0: Normal")
    print("  1: Pathologic Myopia")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("✓ All three models created successfully!")
    print("✓ All models use EfficientNet-B3 backbone (pretrained on ImageNet)")
    print("✓ All models have trainable parameters enabled for fine-tuning")
    print("✓ Models are ready for training or inference")
    
    # Save model architectures (optional)
    print("\n" + "="*60)
    print("Saving Model Architectures")
    print("="*60)
    
    import os
    os.makedirs('model_architectures', exist_ok=True)
    
    # Save model summaries to text files
    for model, name in [(cnn1, 'cnn1_dr'), (cnn2, 'cnn2_glaucoma'), (cnn3, 'cnn3_pm')]:
        filepath = f'model_architectures/{name}_architecture.txt'
        with open(filepath, 'w') as f:
            f.write(str(model))
        print(f"✓ Saved {name} architecture to {filepath}")
    
    print("\n" + "="*60)
    print("Models Created Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your datasets (EyePACS, REFUGE, PALM)")
    print("2. Use the training scripts to train the models:")
    print("   - python train_cnn1_dr.py --data_dir <path> --labels_file <path>")
    print("   - python train_cnn2_glaucoma.py --data_dir <path> --labels_file <path>")
    print("   - python train_cnn3_pm.py --data_dir <path> --labels_file <path>")
    print("3. Or use the models for inference with pre-trained weights")
    print("\n")


if __name__ == '__main__':
    main()
