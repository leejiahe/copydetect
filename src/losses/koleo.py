import torch
import torch.nn as nn

class NTXentKoLeoLoss(nn.Module):
    def __init__(self,
                temperature:float = 0.5,
                entropy_weight:float = 10,
                ):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
    
    def forward(self,
                embeddings,
                labels,
                all_embeddings,
                all_labels,
                rank,
               ) -> torch.Tensor:
        
        N = embeddings.size(0)
        
        similarity = embeddings.matmul(all_embeddings.transpose(0, 1))
        matches = labels.unsqueeze(1) == all_labels.unsqueeze(0)
        non_matches = matches == 0
        
        identity = torch.zeros_like(matches).to(embeddings)
        identity[:, rank * N : (rank + 1) * N] = torch.eye(N).to(embeddings)
        nontrivial_matches = matches * (~identity.bool())
        
        
        small_value = torch.tensor(-100.0).to(embeddings)  # any value > max L2 normalized distance
        
        # NTXent Loss
        logits = (similarity / self.temperature).exp()
        partitions = logits + ((non_matches * logits).sum(dim=1) + 1e-6).unsqueeze(1)
        probabilities = logits / partitions
        ntxent_loss = (-probabilities.log() * nontrivial_matches).sum() / similarity.size(0) # N
        
        # KoLeo Loss
        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(dim = 1, keepdim = True)
        closest_distance = (2 - (2 * max_non_match_sim)).clamp(min=1e-6).sqrt()
        entropy_loss = -closest_distance.log().mean()
        
        # Total Loss
        loss = ntxent_loss + entropy_loss * self.entropy_weight
        
        """
        stats = {'positive_sim': (similarity * nontrivial_matches).sum() / nontrivial_matches.sum(),
                 'negative_sim': (similarity * non_matches).sum() / non_matches.sum(),
                 'nearest_negative_sim': max_non_match_sim.mean(),
                 'center_l2_norm': embeddings.mean(dim = 0).pow(2).sum().sqrt()}
        """
        
        loss_stats = {'ntxent': ntxent_loss,
                      'koleo': entropy_loss} 
        
        return loss, loss_stats