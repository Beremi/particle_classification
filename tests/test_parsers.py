from pathlib import Path

from particle_classification.data.info import parse_info_file
from particle_classification.data.t3pa import count_t3pa_rows, iter_t3pa_hits, matrix_index_to_xy, toa_ftoa_to_time_ns, toa_ftoa_to_time_ticks


FIXTURES = Path(__file__).parent / "fixtures"


def test_matrix_index_to_xy_row_major():
    assert matrix_index_to_xy(0) == (0, 0)
    assert matrix_index_to_xy(255) == (255, 0)
    assert matrix_index_to_xy(256) == (0, 1)
    assert matrix_index_to_xy(511) == (255, 1)


def test_iter_t3pa_hits_and_count():
    path = FIXTURES / "mini.t3pa"
    hits = list(iter_t3pa_hits(path))
    assert count_t3pa_rows(path) == 4
    assert hits[0].x == 0
    assert hits[2].y == 1
    assert hits[-1].tot == 11
    assert hits[-1].ftoa == 3


def test_toa_ftoa_to_fine_timestamp():
    assert toa_ftoa_to_time_ticks(100, 0) == 100.0
    assert toa_ftoa_to_time_ticks(100, 8) == 99.5
    assert toa_ftoa_to_time_ns(100, 16) == 2475.0


def test_parse_info_file():
    metadata = parse_info_file(FIXTURES / "mini.t3pa.info")
    assert metadata["chipboardid"] == "G07-W0050"
    assert metadata["acq_time"] == 20.0
    assert metadata["hv"] == 80.0
    assert metadata["threshold"] == 3.49132
    assert metadata["pixet_version"] == "1.8.0"
