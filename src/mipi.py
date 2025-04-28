import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)

MIPI_CATALOG_URL = (
    "https://api.nationalgas.com/operationaldata/v1/publications/catalogue"
)

MIPI_DATA_URL = "https://api.nationalgas.com/operationaldata/v1/publications/gasday"


def process_subcategories(cat_item: dict, parent_category: str) -> list:
    """
    Recursively processes category and subcategory entries from a catalogue item.
    This function traverses through a nested category structure and extracts catalogue entries,
    maintaining the hierarchical relationship between categories and subcategories.
    Args:
        cat_item (dict): A dictionary containing either catalogueEntries or subCategory items
        parent_category (str): The parent category name, used to build the category hierarchy path
    Returns:
        list: A list of dictionaries containing processed catalogue entries with their following keys:
            - name: The name of the catalogue entry
            - publicationId: The publication ID of the entry
            - category: The immediate category name
            - subcategory: The full category path including parent categories
    """
    result = []
    if "catalogueEntries" in cat_item:
        for entry in cat_item["catalogueEntries"]:
            result.append(
                {
                    "name": entry["name"],
                    "publicationId": entry["publicationId"],
                    "category": cat_item["name"],
                    "subcategory": parent_category,
                }
            )
        return result

    if "subCategory" in cat_item:
        for subcat in cat_item["subCategory"]:
            new_parent = (
                f"{parent_category}/{subcat['name']}"
                if parent_category
                else subcat["name"]
            )
            result.extend(process_subcategories(subcat, new_parent))
        return result


def get_mipi_catalogue_items() -> pd.DataFrame:
    """
    Retrieves and processes MIPI catalogue items from an API endpoint.
    This function fetches the MIPI catalogue data from a predefined URL, processes the
    hierarchical category structure, and converts it into a pandas DataFrame.
    Returns:
        pandas.DataFrame: A DataFrame containing the processed MIPI catalogue items
            with their respective category information.
    Raises:
        requests.exceptions.RequestException: If there's an error fetching data from the API
        KeyError: If the expected 'data' key is not found in the API response
        JSONDecodeError: If the API response cannot be parsed as JSON
    """
    response = requests.get(MIPI_CATALOG_URL)
    if response.status_code != 200:
        logger.error(f"API request failed with status code {response.status_code}. Error: {response.text}")
        return pd.DataFrame()
    
    catalogue = response.json()["data"]
    items = []
    for category in catalogue:
        items.extend(process_subcategories(category, ""))

    return pd.DataFrame(items)


def get_mipi_data(names: list, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Retrieves MIPI data for a specific publication between given dates.

    Args:
        names (list): Names of the publications to fetch
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame containing the publication data with timestamps and values
    """
    # Get catalogue and find publication IDs
    catalogue_df = get_mipi_catalogue_items()
    pub_ids = catalogue_df[catalogue_df["name"].isin(names)]["publicationId"].tolist()

    # Prepare and send request
    payload = {
        "fromDate": from_date,
        "toDate": to_date,
        "publicationIds": pub_ids,
        "latestValue": "Y",
    }

    response = requests.post(MIPI_DATA_URL, json=payload)
    
    if response.status_code != 200:
        logger.error(f"API request failed with status code {response.status_code}. Error: {response.text}")
        return pd.DataFrame()

    data = response.json()
    # Extract publications data and create DataFrame
    result = []
    for item in data:
        pub_name = item["publicationName"]
        for pub in item["publications"]:
            pub["publicationName"] = pub_name
            result.append(pub)

    # for backwards compatibility with the SOAP API all headers are capitalised
    df = pd.DataFrame(result)
    df.columns = [col[0].upper() + col[1:] for col in df.columns]
    return df
