import json


with open("../acl_anthology.json") as f:
    data = json.load(f)

dic = {}
all_abstracts = []
for year in data:
    abstracts = []
    for conf in data[year]:
        for subconf in data[year][conf]:
            for paper in data[year][conf][subconf]["papers"]:
                if "language" in paper and paper["language"] != "eng":
                    continue
                else:
                    if "preprocessed_abstract" not in paper.keys():
                        continue
                    else:
                        abstracts.append(paper["preprocessed_abstract"])
    if abstracts != []:
        with open(f"./data/{year}.txt", "w") as f:
            f.write(" ".join(abstracts).lower())
    all_abstracts += abstracts
    dic[int(year)] = len(abstracts)
with open("./data/compass.txt", "w") as f:
    f.write(" ".join(all_abstracts).lower())

print(*sorted(dic.items(), key=lambda x: x[0]))
