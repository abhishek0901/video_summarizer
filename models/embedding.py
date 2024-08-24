'''
================================================

Embedding model for video data.

Author : Abhishek Srivastava

================================================
'''
import torch
from torchvision import models
from config import CONFIG
from helper_methods.wrappers import measure_execution_time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='tmp/run.log', encoding='utf-8', level=logging.INFO)

class Embedding(torch.nn.Module):
    """
        Description
        ===========
        An Embedding model which internally uses Frozen VGG16 along with a CLS token.
    """

    def __init__(self):
        super(Embedding, self).__init__()
        self.vgg16 = models.vgg16_bn(weights='IMAGENET1K_V1').to(CONFIG.DEVICE) # put model on correct device
        for param in self.vgg16.parameters():
            param.require_grad = False
        D = self.vgg16.classifier[-1].out_features
        self.CLS = torch.rand((3,1,D),device = CONFIG.DEVICE, requires_grad = True)
        logger.info(f"Embedding Model initialized on device : {CONFIG.DEVICE}")
    
    @measure_execution_time
    def forward(self, X):
        """
            X = (T,3,H,W)

            X' = vgg(X) = (T,D)

            X'' = Concat_{axis=0}(X',X',X') = (3,T,D)

            X_{CLS} = (3,1,D)

            E = Concat_{axis=1}(X_{CLS}, X'') = (3,T+1,D)
        """

        T,_,_,_ = X.shape
        D = self.vgg16.classifier[-1].out_features
        assert len(X.shape) == 4

        X_dash = self.vgg16(X).unsqueeze(0) # Converting a 2 dimension array to 3 dimension
        X_double_dash = torch.cat((X_dash, X_dash, X_dash), dim=0)

        assert len(X_double_dash.shape) == 3
        assert X_double_dash.shape == (3,T,D)

        E = torch.cat((self.CLS, X_double_dash), dim=1)

        assert len(E.shape) == 3
        assert E.shape == (3,T+1,D)

        return E

if __name__ == '__main__':
    embd = Embedding()
    X = torch.rand(32,3,150,200, device = CONFIG.DEVICE)
    E = embd(X)
    logger.info(E.shape)
