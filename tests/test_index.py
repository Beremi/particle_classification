from pathlib import Path

from particle_classification.data.index import index_raw_data, write_index_csv, write_index_markdown


FIXTURES = Path(__file__).parent / "fixtures"


def test_index_folder_and_write_outputs(tmp_path):
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "mini.t3pa").write_text((FIXTURES / "mini.t3pa").read_text(), encoding="utf-8")
    (run / "mini.t3pa.info").write_text((FIXTURES / "mini.t3pa.info").read_text(), encoding="utf-8")

    rows = index_raw_data(tmp_path)
    assert len(rows) == 2
    t3pa = next(row for row in rows if row.extension == ".t3pa")
    assert t3pa.rows == 4
    assert t3pa.metadata["chipboardid"] == "G07-W0050"

    csv_path = tmp_path / "index.csv"
    md_path = tmp_path / "index.md"
    write_index_csv(rows, csv_path)
    write_index_markdown(rows, md_path)
    assert "mini.t3pa" in csv_path.read_text(encoding="utf-8")
    assert "Total `.t3pa` hit rows: 4" in md_path.read_text(encoding="utf-8")
