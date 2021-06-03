import os
import time

from twec.twec import TWEC


def train(
    data_dir="./data/",
    embedding_size=300,
    skipgram=False,
    siter=10,
    diter=10,
    negative_samples=10,
    window_size=5,
    output_path="./model",
    overwrite_compass=True,
):
    aligner = TWEC(
        size=embedding_size,
        sg=int(skipgram),
        siter=siter,
        diter=diter,
        workers=4,
        ns=negative_samples,
        window=window_size,
        opath=output_path,
    )
    start = time.time()
    # train the compass: the text should be the concatenation of the text from the slices
    aligner.train_compass(
        os.path.join(data_dir, "compass.txt"), overwrite=overwrite_compass
    )
    # keep an eye on the overwrite behaviour
    end = time.time()
    print("Time Taken for TWEC Pre-Training:", (end - start), " ms")

    slices = {}
    for file in sorted(os.listdir(data_dir)):
        start = time.time()
        slices[file.split(".")[0]] = aligner.train_slice(
            os.path.join(data_dir, file), save=True
        )
        end = time.time()
        print("Time Taken for TWEC Fine-tuning:", (end - start), " ms")


if __name__ == "__main__":
    train()
