# %%
from typing import Optional, Union
import numpy as np
from pydantic import BaseModel


# %%
# ELM model interface

class ELM(BaseModel):
    """Simplified interface of ELM model."""
    
    def __init__(self, 
                inputs: int, 
                outputs: int, 
                classification: str = "", 
                w: Optional[np.ndarray] = None, 
                batch: int = 1000, 
                norm: Optional[float] = None):
        """Initialize an Extreme Learning Machine model.
        
        Args:
            inputs: Dimensionality of input data, or number of data features.
            outputs: Dimensionality of output data, or number of classes.
            classification: Train ELM for classification ('c'), weighted classification ('wc'),
                or multi-label classification ('ml'). Default: "" (regression).
            w: Weights vector for weighted classification, length (outputs * 1).
            batch: Batch size for data processing in ELM, reduces memory requirements.
                Does not work for model structure selection. Default: 1000.
            norm: L2-normalization parameter. None gives the default value.
        """
    
    # Public Methods
    def add_neurons(self, number: int, func: str) -> None:
        """Add neurons to ELM model
        
        Args:
            number: Number of neurons to add
            func: Type of neurons ('lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf')
        """
        pass
    
    def train(self, X: Union[np.ndarray, str], T: Union[np.ndarray, str]):
        """Train ELM model with model structure selection
        
        Args:
            X: Input data matrix (N * inputs)
            T: Outputs data matrix (N * outputs)        
        """
        pass
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs Y for the given input data X
        
        Args:
            X: Input data (N * inputs)
        
        Returns:
            Output data or predicted classes (N * outputs)
        """
        pass
