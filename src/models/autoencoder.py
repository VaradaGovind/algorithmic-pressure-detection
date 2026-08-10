import torch
import torch.nn as nn
import torch.optim as optim

class PhysiologicalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(PhysiologicalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, actual, predicted):
        """
        Calculates MSE Loss for each sample to determine anomaly score.
        A high score indicates deviation from baseline.
        """
        criterion = nn.MSELoss(reduction='none')
        return criterion(predicted, actual).mean(dim=1)


class ImprovedPhysiologicalAutoencoder(nn.Module):
    """Enhanced autoencoder with deeper architecture and better regularization."""
    def __init__(self, input_dim, latent_dim=16, dropout_rate=0.3):
        super(ImprovedPhysiologicalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> 128 -> 64 -> 32 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, latent_dim),
        )
        
        # Decoder: latent_dim -> 32 -> 64 -> 128 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_reconstruction_error(self, actual, predicted):
        """Calculates MSE Loss for each sample to determine anomaly score."""
        criterion = nn.MSELoss(reduction='none')
        return criterion(predicted, actual).mean(dim=1)
    
    def get_latent_representation(self, x):
        """Get the latent representation for a given input."""
        return self.encoder(x)