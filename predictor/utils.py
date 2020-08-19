import torch


def get_mask_for(x):
    # exclude day 1 and padding days
    mask = x.bool().any(dim=-1)
    mask[:, 0] = False
    return mask


def convert_preds(x, y, y_hat):
    # get the indices of the last day of a specific ICU stay
    indices = x.bool().any(dim=-1).int().sum(dim=-1) - 1

    # create a new y and y_hat only considering the values
    # of the last day predictions
    _y = []
    _y_hat = []

    # TODO: somehow refactor this so that it doesn't use a for-loop
    for i in range(y.size(0)):
        last_day_index = indices[i]
        _y.append(y[i, last_day_index, 0])
        _y_hat.append(y_hat[i, last_day_index, 0])

    return (
        torch.tensor(_y, dtype=torch.float32),
        torch.tensor(_y_hat, dtype=torch.float32),
    )
