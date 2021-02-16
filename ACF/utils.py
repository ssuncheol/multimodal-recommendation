import torch



def optimizer(optim, lr, model):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
         
    return optimizer