


def dice_loss(input, target):
    import torch 
    """
        input(X) : predicted sets   -> range[0, 1]
        target(Y): ground truth     -> binary labels {0, 1}
        DSC = 2|X.Y|/(|X|+|Y|)
        X.Y: intersection | true positive | number of overlappig labels
        |X|+|Y|: union
        |X|, |Y|: the number of labels in each set

    """
    # import torch.nn as nn
    # m = nn.Sigmoid()

    eps = 1e-5

    input_ = input.view(-1)
    target_ = target.view(-1)

    intersec = torch.dot(input_, target_).sum()
    union = torch.sum(input_*input_) + torch.sum(target_*target_)
    # union = torch.sum(input_*input_) + torch.sum(target_*target_)
    dsc = (2.0 * intersec+eps)/(union+eps)

    return 1-dsc
