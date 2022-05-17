# Based on the work of ODE LSTM authors Mathias Lechner ad Ramin Hasani
import torch
import torch.utils.data as data

from .p300 import P300Dataset


# TO-DO reformat this. It works, but it's ugly.
def load_dataset(
    batch_size=128,
    data_dir="../data/demons",
):
    """Obtains dataloaders for training diiferent networks on different datasets

    Args:
        ds: dataset to load. Options: activity/p300.
        timestamps: whether to have timestamps in dataloader.
            some architectures need it, some - don't.
        coeffs: whether to have features as raw data or its cubic pline coeffs.
            Needed for Neural CDE.
        irregular: whether to make the dataset irregular by dropping 20% of it's values.
        transpose: if False batch shape is (batch, seq_len, channels),
            if True -- (batch, channels, seq_len)
        batch_size: simply batch size.
        data_dir: directory, where data files are stored.
    """
    dataset = P300Dataset(data_dir=data_dir)
    dataset.get_data_for_experiments(True)

    train_x = torch.Tensor(dataset.train_x)
    test_x = torch.Tensor(dataset.test_x)

    train_y = torch.LongTensor(dataset.train_y)
    test_y = torch.LongTensor(dataset.test_y)

    counts = test_y.unique(return_counts=True)[1].to(torch.float)
    class_balance = counts / counts.min()
    in_features = train_x.size(-1)
    num_classes = int(torch.max(train_y).item() + 1)

    trainloader = data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return (
        trainloader,
        testloader,
        in_features,
        num_classes,
        return_sequences,
        class_balance,
    )
