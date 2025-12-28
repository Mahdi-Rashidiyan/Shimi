"""
JEPA Materials Discovery - Data Pipeline & Baseline Model
Fast implementation for demo - trains in minutes on CPU, hours on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
import requests
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import time
from pymatgen.core import Structure, Lattice
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.local_env import CrystalNN

# ============================================================================
# DATA LOADING - Materials Project API
# ============================================================================

class MaterialsDataset(Dataset):
    """Load crystal structures from Materials Project"""
    
    def __init__(self, root, api_key=None, num_samples=1000):
        self.api_key = api_key
        self.num_samples = num_samples
        self.data_list = []
        super().__init__(root)
        
    @property
    def raw_file_names(self):
        """List of raw file names (required by torch_geometric)"""
        return ['materials_data.json']
    
    @property
    def processed_file_names(self):
        """List of processed file names (required by torch_geometric)"""
        return ['materials_data.pt']
    
    def download(self):
        """Download data from Materials Project API"""
        if self.api_key:
            print(f"🔑 Using API key to download {self.num_samples} materials from Materials Project...")
            self._download_from_mp()
        else:
            print("⚠️  No API key - generating synthetic data for demo")
            self._generate_synthetic_data()
    
    def process(self):
        """Process the raw data (required by torch_geometric)"""
        # In our case, we generate synthetic data directly in download()
        # So this method can be empty
        pass
    
    def _download_from_mp(self):
        """Download real data from Materials Project"""
        try:
            with MPRester(self.api_key) as mpr:
                # Get materials with formation energy and band gap
                # Using the updated API syntax
                print("📡 Querying Materials Project API...")
                
                # First, get a list of material IDs that match our criteria
                docs = mpr.materials.summary.search(
                    formation_energy_per_atom=("lt", 0.5),  # Relatively stable
                    band_gap=("gte", 0),  # Has band gap
                    nsites=("gte", 3, "lte", 50),  # Between 3 and 50 sites
                    fields=["material_id", "formation_energy_per_atom", "band_gap", "structure"],
                    num_chunks=self.num_samples // 1000 + 1,
                    chunk_size=1000
                )
                
                print(f"📊 Found {len(docs)} materials")
                
                # Process the results
                processed_count = 0
                for doc in docs:
                    if processed_count >= self.num_samples:
                        break
                    
                    try:
                        # Extract structure
                        structure = doc.structure
                        
                        # Convert to graph representation
                        data = self._structure_to_data(structure, doc)
                        if data is not None:
                            self.data_list.append(data)
                            processed_count += 1
                            
                            if processed_count % 100 == 0:
                                print(f"   Processed {processed_count}/{self.num_samples} materials")
                    
                    except Exception as e:
                        print(f"⚠️  Error processing material {doc.material_id}: {e}")
                        continue
                
                print(f"✓ Successfully processed {len(self.data_list)} materials")
                
        except Exception as e:
            print(f"❌ Error connecting to Materials Project: {e}")
            print("🔄 Falling back to synthetic data...")
            self._generate_synthetic_data()
    
    def _structure_to_data(self, structure, doc):
        """Convert pymatgen Structure to torch_geometric Data"""
        try:
            # Get crystal structure information
            num_atoms = len(structure)
            
            # Atomic features: [atomic_num, atomic_radius, electronegativity]
            x = []
            for site in structure:
                elem = site.species.elements[0]
                atomic_num = elem.Z
                # Simplified atomic radius (in Angstroms)
                atomic_radius = elem.atomic_radius if elem.atomic_radius else 1.5
                # Simplified electronegativity (Pauling scale)
                electronegativity = elem.X if elem.X else 2.0
                
                x.append([atomic_num, atomic_radius, electronegativity])
            
            x = torch.tensor(x, dtype=torch.float)
            
            # Get positions
            pos = torch.tensor(structure.cart_coords, dtype=torch.float)
            
            # Create edges using CrystalNN for better connectivity
            try:
                cnn = CrystalNN(cutoff=0.5)
                edges = cnn.get_bonded_structure(structure)
                
                edge_index = []
                edge_attr = []
                
                for i, j, edge_data in edges.get_bonds():
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Edge feature: distance
                    dist = structure.get_distance(i, j)
                    edge_attr.append([dist])
                    edge_attr.append([dist])
                
                if len(edge_index) == 0:
                    # Fallback to distance-based edges
                    edge_index, edge_attr = self._create_distance_edges(structure)
                    
            except:
                # Fallback to distance-based edges
                edge_index, edge_attr = self._create_distance_edges(structure)
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Target properties
            formation_energy = doc.formation_energy_per_atom
            band_gap = doc.band_gap
            
            y = torch.tensor([formation_energy, band_gap], dtype=torch.float)
            
            data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr,
                pos=pos, 
                y=y, 
                num_atoms=num_atoms,
                material_id=doc.material_id
            )
            
            return data
            
        except Exception as e:
            print(f"Error converting structure to data: {e}")
            return None
    
    def _create_distance_edges(self, structure, cutoff=5.0):
        """Create edges based on distance cutoff"""
        edge_index = []
        edge_attr = []
        
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                dist = structure.get_distance(i, j)
                if dist < cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append([dist])
                    edge_attr.append([dist])
        
        if len(edge_index) == 0:
            # Ensure at least some connectivity
            edge_index = [[0, 1], [1, 0]]
            edge_attr = [[1.0], [1.0]]
        
        return edge_index, edge_attr
    
    def _generate_synthetic_data(self):
        """Generate synthetic crystal data for demo purposes"""
        print(f"Generating {self.num_samples} synthetic materials...")
        
        # Common elements and their properties
        elements = {
            'Si': {'atomic_num': 14, 'radius': 1.1},
            'O': {'atomic_num': 8, 'radius': 0.66},
            'Fe': {'atomic_num': 26, 'radius': 1.26},
            'Al': {'atomic_num': 13, 'radius': 1.43},
            'Ca': {'atomic_num': 20, 'radius': 1.97},
            'Mg': {'atomic_num': 12, 'radius': 1.60},
        }
        
        for i in range(self.num_samples):
            # Random crystal structure (3-10 atoms)
            num_atoms = np.random.randint(3, 11)
            
            # Random element selection
            atom_types = np.random.choice(list(elements.keys()), num_atoms)
            
            # Atomic features: [atomic_num, radius, electronegativity]
            x = torch.tensor([
                [elements[a]['atomic_num'], 
                 elements[a]['radius'],
                 np.random.rand()]  # simplified electronegativity
                for a in atom_types
            ], dtype=torch.float)
            
            # Random 3D positions (normalized to unit cell)
            pos = torch.rand(num_atoms, 3)
            
            # Create edges based on distance (cutoff = 3.0 Angstroms)
            edge_index = []
            edge_attr = []
            for j in range(num_atoms):
                for k in range(j+1, num_atoms):
                    dist = torch.norm(pos[j] - pos[k]).item()
                    if dist < 0.5:  # normalized distance threshold
                        edge_index.append([j, k])
                        edge_index.append([k, j])
                        edge_attr.append([dist])
                        edge_attr.append([dist])
            
            if len(edge_index) == 0:
                # Ensure at least some connectivity
                edge_index = [[0, 1], [1, 0]]
                edge_attr = [[0.3], [0.3]]
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Target properties (synthetic, physically plausible)
            # Formation energy: roughly correlates with structure
            formation_energy = -2.0 + np.random.randn() * 0.5 - 0.1 * num_atoms
            
            # Band gap: 0-5 eV range
            avg_atomic_num = np.mean([elements[a]['atomic_num'] for a in atom_types])
            band_gap = max(0, 3.0 - 0.1 * avg_atomic_num + np.random.randn() * 0.8)
            
            y = torch.tensor([formation_energy, band_gap], dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                       pos=pos, y=y, num_atoms=num_atoms)
            self.data_list.append(data)
        
        print(f"✓ Generated {len(self.data_list)} synthetic materials")
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


# ============================================================================
# BASELINE MODEL - Crystal Graph Convolutional Network (CGCNN)
# ============================================================================

class CGConv(MessagePassing):
    """Crystal Graph Convolutional Layer"""
    
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.hidden_dim = node_dim  # Renamed to avoid conflict with MessagePassing.node_dim
        self.edge_dim = edge_dim
        
        # Edge network
        self.edge_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + edge_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate node features and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(z)
    
    def update(self, aggr_out, x):
        # Update node features
        z = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(z)


class BaselineCGCNN(nn.Module):
    """Baseline Crystal Graph Convolutional Neural Network"""
    
    def __init__(self, node_features=3, edge_features=1, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            CGConv(hidden_dim, edge_features)
            for _ in range(num_layers)
        ])
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2)  # Predict [formation_energy, band_gap]
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.silu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Predict properties
        out = self.readout(x)
        return out


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_baseline(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    """Train baseline model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print("\n🚀 Training Baseline CGCNN Model...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            
            # Fix: Ensure y has the right shape [batch_size, 2]
            if data.y.dim() == 1:
                batch_size = out.size(0)
                y = data.y.view(batch_size, 2)
            else:
                y = data.y
                
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                
                # Fix: Ensure y has the right shape [batch_size, 2]
                if data.y.dim() == 1:
                    batch_size = out.size(0)
                    y = data.y.view(batch_size, 2)
                else:
                    y = data.y
                    
                loss = F.mse_loss(out, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("=" * 60)
    print("✓ Training Complete!\n")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            
            # Fix: Ensure y has the right shape [batch_size, 2]
            if data.y.dim() == 1:
                batch_size = out.size(0)
                y = data.y.view(batch_size, 2)
            else:
                y = data.y
                
            all_preds.append(out.cpu())
            all_targets.append(y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics for each property
    results = {}
    properties = ['Formation Energy', 'Band Gap']
    
    for i, prop in enumerate(properties):
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        r2 = r2_score(targets[:, i], preds[:, i])
        results[prop] = {'MAE': mae, 'R2': r2}
        print(f"{prop:20s} | MAE: {mae:.4f} | R²: {r2:.4f}")
    
    return results, preds, targets


# ============================================================================
# MAIN DEMO SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JEPA MATERIALS DISCOVERY - BASELINE DEMO")
    print("=" * 60)
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_SAMPLES = 1000
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Get API key from user or environment
    API_KEY = os.environ.get('MP_API_KEY') or input("Enter your Materials Project API key (or press Enter for synthetic data): ").strip()
    
    print(f"\n📊 Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Samples: {NUM_SAMPLES}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Using API: {'Yes' if API_KEY else 'No (synthetic data)'}")
    
    # Create dataset
    print("\n📦 Loading Dataset...")
    
    # Create directory if it doesn't exist
    os.makedirs('./shimi', exist_ok=True)
    
    dataset = MaterialsDataset(root='./shimi', api_key=API_KEY, num_samples=NUM_SAMPLES)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Create and train baseline model
    model = BaselineCGCNN(node_features=3, edge_features=1, hidden_dim=128, num_layers=3)
    train_losses, val_losses = train_baseline(model, train_loader, val_loader, 
                                              epochs=EPOCHS, device=DEVICE)
    
    # Evaluate
    print("\n📈 Test Set Results:")
    print("=" * 60)
    results, preds, targets = evaluate_model(model, test_loader, device=DEVICE)
    
    # Save model
    torch.save(model.state_dict(), 'baseline_cgcnn.pt')
    print("\n💾 Model saved to 'baseline_cgcnn.pt'")
    
    print("\n✅ Baseline complete! Ready for JEPA comparison.")