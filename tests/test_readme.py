import pandas as pd

from emailexp.readme import update_readme


def test_readme_block_is_replaced_and_rest_kept(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- RESULTS:START -->\nPLACEHOLDER\n<!-- RESULTS:END -->\n\nFooter $1 \\1\n"
    )
    effects = pd.DataFrame(
        [
            {
                "arm": "Men's email",
                "outcome": "spend",
                "lift": 0.77,
                "ci_low": 0.49,
                "ci_high": 1.05,
                "relative_lift": 0.7,
                "p_holm": 0.0001,
            }
        ]
    )
    policies = pd.DataFrame(
        [
            {"policy": "Targeted: X", "incremental_profit": 22_900, "ci_low": 10_000, "ci_high": 35_000},
            {
                "policy": "Men's email to everyone",
                "incremental_profit": 20_800,
                "ci_low": 9_000,
                "ci_high": 31_000,
            },
        ]
    )
    qini = pd.DataFrame(
        [
            {"model": m, "arm": a, "outcome": "spend", "qini": q}
            for m, q in (("X", 0.03), ("Random", -0.01))
            for a in (1, 2)
        ]
    )
    results = {
        "n_customers": 64_000,
        "srm_p": 0.9,
        "max_abs_smd": 0.005,
        "champion": "X",
        "margin": 0.4,
        "email_cost": 0.1,
    }
    update_readme(readme, results, effects, policies, qini)
    text = readme.read_text()
    assert "PLACEHOLDER" not in text and "+$0.77" in text and "$22,900" in text
    assert text.startswith("# Title") and text.endswith("Footer $1 \\1\n")
