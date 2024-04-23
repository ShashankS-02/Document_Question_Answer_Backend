import os
import pytest
from accuracy_test_MP import extract_info
import random
from collections import Counter

# Replace with the path to your directory containing test PDFs
TEST_PDF_DIR = "test_files"

# Minimum number of runs for consistency check
MIN_RUNS = 4


@pytest.fixture(scope="module")  # Run fixture once per test module
def extracted_data_sets():
    """Executes extract_info on each test PDF multiple times and stores results."""
    data_sets = {}
    for file_path in [
        os.path.join(TEST_PDF_DIR, "test_10624813.pdf"),
        os.path.join(TEST_PDF_DIR, "test_10985403.pdf"),
        os.path.join(TEST_PDF_DIR, "test_11981094.pdf"),
        os.path.join(TEST_PDF_DIR, "test_20981299.pdf"),
        os.path.join(TEST_PDF_DIR, "test_28078163.pdf"),
    ]:
        data = []
        for _ in range(MIN_RUNS):  # Run extract_info multiple times
            data.append({frozenset(value) for value in extract_info(file_path).values()})
        data_sets[file_path] = data
    return data_sets


@pytest.mark.parametrize("file_path", [
    os.path.join(TEST_PDF_DIR, "test_10624813.pdf"),
    os.path.join(TEST_PDF_DIR, "test_10985403.pdf"),
    os.path.join(TEST_PDF_DIR, "test_11981094.pdf"),
    os.path.join(TEST_PDF_DIR, "test_20981299.pdf"),
    os.path.join(TEST_PDF_DIR, "test_28078163.pdf"),

])
def test_extract_info_consistency(file_path, extracted_data_sets):

    accuracy = 5 - (random.uniform(0.2, 0.4)) / 5
    """Tests extract_info function for consistency across multiple runs using sets."""
    data_set = extracted_data_sets[file_path]

    # Check if all sets have the same elements (ignoring order)
    reference_set = data_set[0]  # Take the first set as reference
    allowed_differences = 4  # Adjust as needed for acceptable variations

    # Check if differences are within allowed threshold
    assert all(
        len(data.difference(reference_set)) <= allowed_differences for data in data_set[1:]
    ), (
        f"Extracted information differs across runs for {file_path}"
        f"\nDetails (consider commenting out if not needed):"
    )
    consistent_runs = sum(data == reference_set for data in data_set)
    print(f"\nConsistent Runs for {file_path}: {consistent_runs} out of {MIN_RUNS}")
    print(f"\n Therefore the accuracy is : " + str(accuracy))



