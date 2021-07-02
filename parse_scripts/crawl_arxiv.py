import argparse
import json

import arxiv
from tqdm.auto import tqdm


def crawl_arxiv(arxiv_query, max_results, json_save_path):
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = {}
    for result in tqdm(search.get()):
        try:
            year = str(result.published).split("-")[0]

            if year not in papers:
                papers[year] = []

            papers[year].append(
                {
                    "url": str(result.entry_id),
                    "date": str(result.published),
                    "title": str(result.title),
                    "authors": [str(author) for author in result.authors],
                    "abstract": str(result.summary),
                    "journal ref": str(result.journal_ref),
                    "category": str(result.primary_category),
                }
            )
        except:
            continue

    with open(json_save_path, "w") as f:
        json.dump(papers, f, indent=4)

    print("Year-wise Distribution:")
    for year in papers:
        print(year, ":", len(papers[year]), "papers")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arxiv_query",
        type=str,
        default="cat:cs.CL",
        help="Search Query for arXiv (check https://arxiv.org/help/api/user-manual)",
    )
    parser.add_argument(
        "--max_results", type=int, default=100000, help="Maximum Papers to Crawl"
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="data/arxiv_anthology.json",
        help=".json path where the file is to saved",
    )
    args = parser.parse_args()

    crawl_arxiv(args.arxiv_query, args.max_results, args.json_save_path)


if __name__ == "__main__":
    main()
