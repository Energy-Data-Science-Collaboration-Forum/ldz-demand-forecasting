import requests
import pandas as pd
from pandas.testing import assert_frame_equal
from src.mipi import process_subcategories, get_mipi_catalogue_items


def test_process_subcategories_direct_entries():
    cat_item_direct = {
        "name": "TestCategory",
        "catalogueEntries": [
            {"name": "Entry1", "publicationId": "pub1"},
            {"name": "Entry2", "publicationId": "pub2"},
        ],
    }
    result_direct = process_subcategories(cat_item_direct, "ParentCat")
    assert len(result_direct) == 2
    assert result_direct[0] == {
        "name": "Entry1",
        "publicationId": "pub1",
        "category": "TestCategory",
        "subcategory": "ParentCat",
    }
    assert result_direct[1] == {
        "name": "Entry2",
        "publicationId": "pub2",
        "category": "TestCategory",
        "subcategory": "ParentCat",
    }


def test_process_subcategories_nested_structure():
    cat_item_sub = {
        "name": "MainCategory",
        "subCategory": [
            {
                "name": "SubCategory",
                "catalogueEntries": [{"name": "SubEntry1", "publicationId": "pub3"}],
            }
        ],
    }
    result_sub = process_subcategories(cat_item_sub, "")
    assert len(result_sub) == 1
    assert result_sub[0] == {
        "name": "SubEntry1",
        "publicationId": "pub3",
        "category": "SubCategory",
        "subcategory": "SubCategory",
    }


def test_get_mipi_catalogue_items(monkeypatch):
    mock_response = {
        "data": [
            {
                "name": "Category1",
                "catalogueEntries": [
                    {"name": "Entry1", "publicationId": "pub1"}
                ]
            },
            {
                "name": "Category2",
                "subCategory": [
                    {
                        "name": "SubCat",
                        "catalogueEntries": [
                            {"name": "Entry2", "publicationId": "pub2"}
                        ]
                    }
                ]
            }
        ]
    }

    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return mock_response

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    result = get_mipi_catalogue_items()
    expected_df = pd.DataFrame([
        {
            "name": "Entry1",
            "publicationId": "pub1",
            "category": "Category1",
            "subcategory": ""
        },
        {
            "name": "Entry2",
            "publicationId": "pub2",
            "category": "SubCat",
            "subcategory": "SubCat"
        }
    ])
    
    assert_frame_equal(result, expected_df)
