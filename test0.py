import torch
import pawner

pawner.attacks(
    torch.ones(1,66,dtype=torch.long).to("cuda:0"), 
    torch.ones(1,dtype=torch.long).to("cuda:0"), 
    attacks := torch.zeros(1,64,dtype=torch.long).to("cuda:0")
)

print(attacks.view(8,8))
