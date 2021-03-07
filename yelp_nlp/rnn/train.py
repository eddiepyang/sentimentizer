import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
from data import id_to_glove, CorpusData
from model import RNN


def fit(
    model: RNN,
    loss_function: object,
    opitmizer: object,
    dataclass: CorpusData,
    batch_size: int,
    epochs: int,
    device: str,
    packed: bool = False
):

    start = time.time()
    model.train()

    epoch_count = 0
    losses = []
    train_loader = DataLoader(dataclass, batch_size=batch_size)

    def train_epoch():

        nonlocal dataclass, train_loader, losses

        i = 0
        n = len(dataclass)

        for j, (sent, target) in enumerate(train_loader):

            optimizer.zero_grad()
            sent, labels = sent.long().to(device), target.float().to(device)
            log_probs = model(sent)
            loss = loss_function(log_probs, labels.to(device))

            # gets graident
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.3,
                norm_type=2
            )

            # updates with new gradient
            optimizer.step()

            i += len(labels)
            losses.append(loss.item())
            if i % (batch_size*100) == 0:
                print(f"""{i/n:.2f} of rows completed in {j+1} cycles, current loss at {np.mean(losses[-30:]):.4f}""")  # noqa: E501

    print('fitting model...')

    for epoch in range(epochs):

        train_epoch()

        epoch_count += 1
        print(f'epoch {epoch_count} complete')
    print(f'fit complete {time.time()-start:.0f} seconds passed')
    return losses


if __name__ == '__main__':

    fpath = '../../data/archive.zip'
    fname = 'yelp_academic_dataset_review.json'
    abs_path = '/projects/yelp_nlp/data/'
    state_path = '../../data/model_weight.pt'
    device = 'cuda'
    batch_size = 200
    input_len = 200
    n_epochs = 2

    dataset = CorpusData(
        fpath=fpath,
        fname=fname,
        stop=100000
    )
    embedding_matrix = id_to_glove(dataset.dict_yelp, abs_path)

    emb_t = torch.from_numpy(embedding_matrix)
    emb = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
    emb.load_state_dict({'weight': emb_t})
    model = RNN(emb_weights=emb_t, batch_size=batch_size, input_len=input_len)
    model.load_weights()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.7, 0.99),
        weight_decay=1e-5
    )
    loss_function = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        100,
        eta_min=0,
        last_epoch=-1
    )
    losses = fit(
        model,
        loss_function,
        optimizer,
        dataset,
        batch_size,
        n_epochs,
        device=device,
        packed=False
    )
    torch.save(model.state_dict(), state_path)
