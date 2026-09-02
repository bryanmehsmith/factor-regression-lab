from html.parser import HTMLParser
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = PROJECT_ROOT / "frontend" / "index.html"
DATA_DIR = PROJECT_ROOT / "frontend" / "js" / "data"


class DataSourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self.depth:
            self.depth += 1
        elif attributes.get("data-data-source") == "synthetic":
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def test_public_page_labels_bundled_regression_data_as_synthetic():
    parser = DataSourceParser()
    parser.feed(FRONTEND_PATH.read_text())
    visible_text = " ".join(" ".join(parser.parts).split())

    assert "deterministic synthetic factors and industry returns" in visible_text
    assert "live refresh" in visible_text
    assert "teaching regression mechanics" in visible_text


def test_bundled_factor_and_industry_series_are_synthetic_and_aligned():
    factors = json.loads((DATA_DIR / "factors.json").read_text())
    industries = json.loads((DATA_DIR / "industries.json").read_text())

    assert factors["metadata"]["kind"] == "synthetic"
    assert industries["metadata"]["kind"] == "synthetic"
    assert len(factors["dates"]) >= 120
    assert factors["dates"] == industries["dates"]
    assert all(len(values) == len(factors["dates"]) for values in factors["series"].values())
    assert all(len(values) == len(industries["dates"]) for values in industries["series"].values())
