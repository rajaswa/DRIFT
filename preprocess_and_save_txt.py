import json
import os

from src.utils.preprocess import preprocess_text


def preprocess_and_save(
    json_path="../acl_anthology.json", text_key="abstract", save_dir="./data"
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(json_path) as f:
        data = json.load(f)
    info_dic = {}
    all_contents = []
    for year in data:
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
                f.write(" ".join(contents).lower())
        all_contents += contents
        info_dic[int(year)] = len(contents)
    with open(os.path.join(save_dir, "compass.txt"), "w") as f:
        f.write("\n".join(all_contents).lower())

    return info_dic


if __name__ == "__main__":
    info_dic = preprocess_and_save()
    print(*sorted(info_dic.items(), key=lambda x: x[0]))
