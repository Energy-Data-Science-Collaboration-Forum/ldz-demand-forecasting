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
