from yelp_nlp.rnn.data import CorpusDataset
from yelp_nlp.rnn.test import Trainer
from yelp_nlp.logging_utils import new_logger

logger = new_logger(20)


def main(
    df_path=None,
    dictionary_path=None,
):

    import argparse
    from yelp_nlp.rnn.config import OptParams, SchedulerParams, loss_function

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive_name",
        default="archive.zip",
        help="file where yelp data is saved, expects an archive of json files",
    )
    parser.add_argument("--fname", default="yelp_academic_dataset_review.json")
    parser.add_argument(
        "--abs_path",
        default="projects/yelp_nlp/data/",
        help="folder where data is stored, path after /home/{user}/",
    )
    parser.add_argument(
        "--state_path",
        default="model_weight.pt",
        help="file name for saved pytorch model weights",
    )
    parser.add_argument(
        "--device", default="cpu", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--input_len", type=int, default=200, help="width of lstm layer"
    )  # noqa: E501
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument(
        "--stop", type=int, default=10000, help="how many lines to load"
    )

    args = parser.parse_args()

    if df_path:

        df = pd.read_parquet(df_path)  # noqa: E501

        dataset = CorpusDataset(
            fpath=os.path.join(args.abs_path, args.archive_name),
            fname=args.fname,
            df=df,
            stop=args.stop,
            max_len=args.input_len,
        )
    else:

        dataset = CorpusDataset(
            fpath=os.path.join(args.abs_path, args.archive_name),
            fname=args.fname,
            stop=args.stop,
            max_len=args.input_len,
        )

    embedding_matrix = id_to_glove(dataset.dict_yelp, args.abs_path)
    emb_t = torch.from_numpy(embedding_matrix)

    model = RNN(emb_weights=emb_t, batch_size=args.batch_size, input_len=args.input_len)

    model.load_weights()

    params = OptParams()

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=params.betas,
        weight_decay=params.weight_decay,
    )

    sp = SchedulerParams()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sp.T_max, eta_min=sp.eta_min, last_epoch=sp.last_epoch
    )

    trainer = Trainer(
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        dataclass=dataset,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        workers=4,
        device=args.device,
        mode="training",
    )

    trainer.fit(model)

    weight_path = os.path.join(os.path.expanduser("~"), args.abs_path, args.state_path)
    torch.save(model.state_dict(), weight_path)
    print(f"model weights saved to: {args.abs_path}{args.state_path}")


if __name__ == "__main__":

    main()
