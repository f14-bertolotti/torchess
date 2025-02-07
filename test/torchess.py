import unittest
import torchess
import torch

class Suite(unittest.TestCase): 

    def test_1(self):

        board = torch.tensor([
            10,  8,  9, 11, 12,  9,  8, 10,  
            7 ,  7,  7,  7,  7,  7,  7,  7,  
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            1 ,  1,  1,  1,  1,  1,  1,  1,  
            4 ,  2,  3,  5,  6,  3,  2,  4,  
            0 ,  0,  0,  0,  0,  0,  0,  0,
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            0 ,  0,  0,  0,  0,  0,  0,  0,  
            0 ,  0,  7,  4,  0,  4,  0,  0,  
            0 ,  0,  0,  0], device='cuda:0', dtype=torch.int32).view(100,1)

        action = torch.tensor([6, 2, 5, 0, 2], device='cuda:0', dtype=torch.int32).view(5,1)
        reward = torch.zeros(2,1, device='cuda:0', dtype=torch.float32)
        done   = torch.zeros(1, device='cuda:0', dtype=torch.bool)


        self.assertTrue((torch.tensor([[0.],[0.]], device='cuda:0') == reward).all())
        self.assertTrue((torch.tensor([False], device='cuda:0') == done).all())
