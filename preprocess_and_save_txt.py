import json
import os

from tqdm.auto import tqdm

from src.utils.preprocess import preprocess_text


def preprocess_and_save(
    json_path="./unprocessed.json",
    text_key="abstract",
    save_dir="./data",
    streamlit=False,
    component=None,
):
    if not json_path.endswith(".json"):
        raise ValueError("Selected `json_path` should end with `.json`.")
    if streamlit and component is None:
        raise ValueError("`component` cannot be `None` when `streamlit` is `True`.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(json_path) as f:
        data = json.load(f)
    info_dic = {}
    all_contents = []
    if streamlit:
        component.write("Preprocessing")
        progress_bar = component.progress(0.0)
        text_output = component.empty()
    for idx, year in enumerate(tqdm(data)):
        if streamlit:
            text_output.info(f"Preprocessing Year {year}")
        contents = []
        for paper in data[year]:
            if "language" in paper and paper["language"] != "eng":
                continue
            else:
                if text_key not in paper.keys():
                    continue
                else:
                    contents.append(preprocess_text(paper[text_key]))
        if contents != []:
            with open(os.path.join(save_dir, f"{year}.txt"), "w") as f:
                f.write("\n".join(contents).lower())
        all_contents += contents
        info_dic[int(year)] = len(contents)
        if streamlit:
            progress = (idx + 1) / len(data)
            if progress > 1.0:
                progress = 1.0
            progress_bar.progress(progress)
    with open(os.path.join(save_dir, "compass.txt"), "w") as f:
        if streamlit:
            text_output.info(f"Saving Compass")
        f.write("\n".join(all_contents).lower())

    return info_dic


if __name__ == "__main__":
    info_dic = preprocess_and_save()
    print(*sorted(info_dic.items(), key=lambda x: x[0]))
