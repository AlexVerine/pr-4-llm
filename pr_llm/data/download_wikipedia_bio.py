import json
import os
from argparse import ArgumentParser

import datasets
import wikipediaapi
from dotenv import load_dotenv
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

from pr_llm.utils import get_env

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--min_content_length", default=80, type=int)
    parser.add_argument("--max_content_length", default=350, type=int)
    load_dotenv()
    args = parser.parse_args()

    # SPARQL query to get the list of human individuals with Wikipedia sitelinks
    sparql_query = """
    PREFIX schema: <http://schema.org/>
    SELECT DISTINCT ?item ?itemLabel ?birthDate ?gender ?genderLabel 
                (GROUP_CONCAT(DISTINCT ?occupationLabel; separator=", ") AS ?occupations) 
                ?wikiTitle
    WHERE {
    ?featuredArticle schema:about ?item.
    ?featuredArticle schema:inLanguage "en".
    ?featuredArticle wikibase:badge ?badge.
    ?item wdt:P31 wd:Q5;  # Instance of human
            wdt:P569 ?birthDate;  # Birth date
            wdt:P21 ?gender.  # Gender
    VALUES (?badge) {(wd:Q17437796)(wd:Q17437798)}
    OPTIONAL {?featuredArticle schema:about ?item;
                                schema:inLanguage "en";
                                schema:name ?wikiTitle.}
    OPTIONAL {?item wdt:P106 ?occupation.
                ?occupation rdfs:label ?occupationLabel.
                FILTER(LANG(?occupationLabel) = "en")}
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    GROUP BY ?item ?itemLabel ?birthDate ?gender ?genderLabel ?wikiTitle
    ORDER BY ?itemLabel
    """

    sparql_endpoint = "https://query.wikidata.org/sparql"

    # Function to execute SPARQL query
    def execute_sparql_query(query):
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    # Function to save results as JSON
    def save_results_to_json(results, filename="wikidata_results.json"):
        with open(filename, "w") as json_file:
            json.dump(results, json_file)

    # Function to load results from JSON
    def load_results_from_json(filename="wikidata_results.json"):
        if os.path.exists(filename):
            with open(filename, "r") as json_file:
                return json.load(json_file)
        return None

    # Initialize Wikipedia API with a proper user agent
    wiki_wiki = wikipediaapi.Wikipedia(user_agent="BiographyDataExtraction/1.0")

    # Check if results file exists, if not, execute the SPARQL query and save results
    results_filename = "wikidata_results.json"
    if not os.path.exists(results_filename):
        results = execute_sparql_query(sparql_query)
        save_results_to_json(results, results_filename)
    else:
        results = load_results_from_json(results_filename)

    # Create a list to hold the dataset samples
    dataset_samples = {
        "title": [],
        "content": [],
        "gender": [],
        "birth_date": [],
        "occupations": [],
    }

    # Iterate through the results and fetch the Wikipedia content
    title_set = set()
    DATA_PATH = get_env("DATA_PATH")
    save_path = DATA_PATH / "wikipedia_bio"
    if not os.path.exists(save_path):
        for result in tqdm(results["results"]["bindings"]):
            item_label = result["itemLabel"]["value"]
            if "wikiTitle" in result:
                wiki_title = result["wikiTitle"]["value"]
                if wiki_title in title_set:
                    continue
                title_set.add(wiki_title)

                page_py = wiki_wiki.page(wiki_title)

                if page_py.exists():
                    # Assume you want to extract the summary (intro) instead of the first section
                    # as it might not be consistently structured across all pages
                    summary_content = page_py.summary

                    dataset_samples["title"].append(item_label)
                    dataset_samples["content"].append(summary_content)

                    # Add the other fields to the dataset
                    dataset_samples["gender"].append(result["genderLabel"]["value"])
                    dataset_samples["birth_date"].append(result["birthDate"]["value"])
                    dataset_samples["occupations"].append(
                        result["occupations"]["value"]
                    )

                else:
                    print(f"Page not found for Wikipedia title: {wiki_title}")
            else:
                print(f"No Wikipedia sitelink found for: {item_label}")

        # Convert the list of samples to a Hugging Face dataset
        hf_dataset = datasets.Dataset.from_dict(dataset_samples)

        # Save the dataset
        hf_dataset.save_to_disk(f"{save_path}_raw")

        print("Hugging Face dataset created and saved.")
    else:
        print(f"Dataset already exists at {save_path}")
        hf_dataset = datasets.load_from_disk(str(save_path))

    # Make a selection based on content length
    hf_dataset = hf_dataset.filter(
        lambda x: args.min_content_length
        <= len(x["content"].split(" "))
        <= args.max_content_length
    )

    hf_dataset.save_to_disk(f"{save_path}")
