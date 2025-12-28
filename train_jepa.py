"""
Complete JEPA Training Script
This script trains the JEPA model and saves it as jepamodel.pt
"""

import torch
import os
from torch_geometric.data import DataLoader

# Import from your existing files
from data_pipeline import MaterialsDataset
from architecture import JEPAMaterials, train_jepa, evaluate_jepa

def main():
    print("\n" + "=" * 80)
    print("JEPA MATERIALS DISCOVERY - TRAINING SCRIPT")
    print("=" * 80)
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_SAMPLES = 1000
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    # Get API key
    API_KEY = os.environ.get('MP_API_KEY') or input("Enter Materials Project API key (or press Enter for synthetic): ").strip()
    API_KEY = API_KEY if API_KEY else None
    
    print(f"\n📊 Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Samples: {NUM_SAMPLES}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Data Source: {'Materials Project' if API_KEY else 'Synthetic'}")
    
    # Create dataset directory
    os.makedirs('./shimi', exist_ok=True)
    
    # Load or create dataset
    print("\n📦 Loading Dataset...")
    dataset = MaterialsDataset(root='./shimi', api_key=API_KEY, num_samples=NUM_SAMPLES)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty! Generating synthetic data...")
        dataset._generate_synthetic_data()
    
    print(f"✓ Loaded {len(dataset)} materials")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"\n📂 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create JEPA model
    print("\n🏗️  Creating JEPA Model...")
    model = JEPAMaterials(
        node_features=3,
        edge_features=1,
        hidden_dim=128,
        latent_dim=32,  # Using 32 as in your configuration
        num_layers=3,
        num_properties=2
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🚀 Starting JEPA Training...")
    print("=" * 80)
    
    history = train_jepa(
        model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=DEVICE
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on Test Set...")
    results, preds, targets, embeddings = evaluate_jepa(model, test_loader, device=DEVICE)
    
    # Save model with configuration
    print("\n💾 Saving JEPA Model...")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_features': 3,
            'edge_features': 1,
            'hidden_dim': 128,
            'latent_dim': 32,
            'num_layers': 3,
            'num_properties': 2
        },
        'results': results,
        'history': history,
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_samples': NUM_SAMPLES,
            'device': DEVICE
        }
    }
    
    torch.save(save_dict, 'jepamodel.pt')
    print("✓ Model saved to 'jepamodel.pt'")
    
    # Also save embeddings for visualization
    torch.save({
        'embeddings': embeddings,
        'targets': targets,
        'predictions': preds
    }, 'jepa_embeddings.pt')
    print("✓ Embeddings saved to 'jepa_embeddings.pt'")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    
    print("\n📊 Final Test Results:")
    for prop, metrics in results.items():
        print(f"   {prop:20s} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
    
    print("\n📁 Generated Files:")
    print("   ✓ jepamodel.pt          - Trained JEPA model")
    print("   ✓ jepa_embeddings.pt    - Embeddings for visualization")
    print("   ✓ baseline_cgcnn.pt     - Baseline model (if previously trained)")
    
    print("\n🎯 Next Steps:")
    print("   1. Run demo_notebook.py to generate visualizations")
    print("   2. Compare with baseline results")
    print("   3. Present to professors!")
    
    print("\n" + "=" * 80 + "\n")
    
    return model, results, history


if __name__ == "__main__":
    try:
        model, results, history = main()
        print("✅ All done! JEPA model is ready.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nTip: Make sure you have data_pipeline.py and architecture.py in the same directory")