import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--dir_path",
        action="store",
        help="Path to dir contains tables to verify"
    )

@pytest.fixture
def dir_path(request):
    return request.config.getoption("--dir_path")