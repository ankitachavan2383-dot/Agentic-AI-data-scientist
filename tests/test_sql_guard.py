import pytest

from app.tools.sql_tools import SqlGuardError, validate_select

T = {"run_1_data"}


def test_allows_select_and_cte():
    validate_select("SELECT a, COUNT(*) FROM run_1_data GROUP BY a", T)
    validate_select("WITH x AS (SELECT * FROM run_1_data) SELECT * FROM x", T)
    validate_select("SELECT EXTRACT(year FROM d) FROM run_1_data", T)


@pytest.mark.parametrize("bad", [
    "DROP TABLE run_1_data", "SELECT 1; DELETE FROM run_1_data", "UPDATE run_1_data SET a=1",
    "SELECT * FROM datasets", "SELECT * FROM pg_user", "SELECT * INTO x FROM run_1_data",
    "SELECT * FROM run_1_data JOIN runs ON 1=1",
])
def test_rejects(bad):
    with pytest.raises(SqlGuardError):
        validate_select(bad, T)
