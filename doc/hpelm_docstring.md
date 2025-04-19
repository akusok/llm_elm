Interface for training Extreme Learning Machines (ELM).

Args:
    inputs (int): dimensionality of input data, or number of data features
    outputs (int): dimensionality of output data, or number of classes
    classification ('c'/'wc'/'ml', optional): train ELM for classfication ('c') / weighted classification ('wc') /
        multi-label classification ('ml'). For weighted classification you can provide weights in `w`. ELM will
        compute and use the corresponding classification error instead of Mean Squared Error.
    w (vector, optional): weights vector for weighted classification, lenght (`outputs` * 1).
    batch (int, optional): batch size for data processing in ELM, reduces memory requirements. Does not work
        for model structure selection (validation, cross-validation, Leave-One-Out). Can be changed later directly
        as a class attribute.
    accelerator ("GPU"/"basic", optional): type of accelerated ELM to use: None, 'GPU', 'basic', ...
    precision (optional): data precision to use, supports single ('single', '32' or numpy.float32) or double
        ('double', '64' or numpy.float64). Single precision is faster but may cause numerical errors. Majority
        of GPUs work in single precision. Default: **double**.
    norm (double, optinal): L2-normalization parameter, **None** gives the default value.
    tprint (int, optional): ELM reports its progess every `tprint` seconds or after every batch,
        whatever takes longer.

Class attributes; attributes that simply store initialization or `train()` parameters are omitted.

Attributes:
    nnet (object): Implementation of neural network with computational methods, but without
        complex logic. Different implementations are given by different classes: for Python, for GPU, etc.
        See ``hpelm.nnets`` folder for particular files. You can implement your own computational algorithm
        by inheriting from ``hpelm.nnets.SLFN`` and overwriting some methods.
    flist (list of strings): Available types of neurons, use them when adding new neurons.

Note:
    Below the 'matrix' type means a 2-dimensional Numpy.ndarray.
