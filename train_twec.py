import os
import time

import numpy as np
import streamlit as st

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
    streamlit=False,
    component=None,
):
    if streamlit and component is None:
        raise ValueError("`component` cannot be `None` when `streamlit` is `True`.")

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

    if streamlit:
        component.write("Training")
        progress = 0.0
        progress_bar = component.progress(progress)
        output = component.beta_expander("Output")

    all_files = sorted(os.listdir(data_dir))
    num_files = len(all_files)
    start = time.time()
    # train the compass: the text should be the concatenation of the text from the slices
    aligner.train_compass(
        os.path.join(data_dir, "compass.txt"), overwrite=overwrite_compass
    )
    # keep an eye on the overwrite behaviour
    end = time.time()
    compass_out = f"Time Taken for TWEC Pre-Training: {(end - start)} ms"
    if not streamlit:
        print(compass_out)
    else:
        progress += 1 / num_files
        progress_bar.progress(np.round(progress, decimals=1))
        with output:
            st.write(compass_out)

    slices = {}
    for file in all_files:
        if file != "compass.txt":
            start = time.time()
            slices[file.split(".")[0]] = aligner.train_slice(
                os.path.join(data_dir, file), save=True
            )
            end = time.time()
            year_out = f"Time Taken for TWEC Fine-tuning for {file.split('.')[0]}: {(end - start)} ms"
            if not streamlit:
                print(year_out)
            else:
                progress += 1 / num_files
                if progress > 1.0:
                    progress = 1.0
                progress_bar.progress(progress)
                with output:
                    st.write(year_out)


if __name__ == "__main__":
    train()
