from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import util.utils as ut
import config as conf
from model.CNNs import FineTuneModel_Hierarchical, FineTuneModel_Hierarchical_Attention
from torch.autograd import Variable
import matplotlib.pyplot as plt

def load_model_auto(model_path, arch='resnet18'):
    """
    Automatically detect and load the correct model type
    (standard or attention-based)
    """
    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=False
    )
    
    # Get the base pretrained model
    base_model = models.__dict__[arch](pretrained=True)
    
    # Try to detect if it's an attention model by checking state_dict keys
    state_dict_keys = checkpoint['state_dict'].keys()
    is_attention = any('cbam' in key or 'channel_attention' in key or 'spatial_attention' in key 
                       for key in state_dict_keys)
    
    # Load appropriate model
    if is_attention:
        print("Detected Attention ResNet18 model")
        model = FineTuneModel_Hierarchical_Attention(base_model, arch, None)
    else:
        print("Detected Standard Hierarchical model")
        model = FineTuneModel_Hierarchical(base_model, arch, None)
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.args = checkpoint['args']
    model.eval()
    
    return model, is_attention


if __name__ == '__main__':
    # Determine model path
    model_name = 'resnet18'
    model_path = os.path.join(
        conf.OUTPUT_WEIGHT_PATH, 
        f'best_{model_name}.pth.tar'
    )
    
    # Auto-load model (detects attention or standard)
    model, is_attention = load_model_auto(model_path, arch=model_name)
    
    print(f"Model type: {'Attention ResNet18' if is_attention else 'Standard ResNet18'}")
    
    # Setup transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    valid_trans = transforms.Compose([
        ut.make_square,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Data loader for training
    dset_train = ImageFolder(root=conf.TRAIN_DIR, transform=valid_trans)
    train_loader = DataLoader(
        dset_train,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=0
    )
    
    # Extract features
    X_train, y_train = [], []
    
    print("Extracting features for t-SNE visualization...")
    
    with torch.no_grad():  # No gradients needed for feature extraction
        for batch_idx, (inputs, y) in enumerate(train_loader):
            input_var = Variable(inputs)
            batch_size = inputs.size(0)
            
            # Forward pass to get features
            outputs = model(input_var)
            
            X_train.append(outputs.data.numpy())
            y_train.extend(y.data.numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * 4} samples...")
    
    print(f"Total samples processed: {len(y_train)}")
    
    # Stack features
    imgs = np.vstack(X_train)
    
    # Apply t-SNE (FIXED: changed n_iter to max_iter)
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=1234, perplexity=30, max_iter=1000)
    X = tsne.fit_transform(imgs)
    
    # Color coding for species
    cl_coding = ['green', 'cyan', 'blue', 'red', 'pink', 'orange']
    color_map = {
        model.args['idx_to_lab'][i]: cl_coding[i]
        for i in range(len(cl_coding))
    }
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Otolith type'] = np.array([model.args['idx_to_lab'][i] for i in y_train])
    
    # Create visualization
    print("Creating t-SNE visualization...")
    plt.figure(figsize=(10, 8))
    axes = sns.scatterplot(
        x="Feature 1",
        y="Feature 2",
        hue="Otolith type",
        data=df,
        palette=color_map,
        s=50,
        alpha=0.7
    )
    
    plt.title(f't-SNE Visualization - {"Attention ResNet18" if is_attention else "Standard ResNet18"}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure
    output_name = 'tSNE_trained_attention.png' if is_attention else 'tSNE_trained.png'
    output_path = os.path.join(conf.OUTPUT_WEIGHT_PATH, output_name)
    
    plt.savefig(
        output_path,
        format='png',
        bbox_inches='tight',
        dpi=500
    )
    print(f"t-SNE plot saved to: {output_path}")
    
    # Save CSV
    csv_name = 'tsne_map_attention.csv' if is_attention else 'tsne_map.csv'
    csv_path = os.path.join(conf.OUTPUT_WEIGHT_PATH, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"t-SNE data saved to: {csv_path}")
    
    plt.close()
    print("✓ t-SNE visualization completed successfully!")
