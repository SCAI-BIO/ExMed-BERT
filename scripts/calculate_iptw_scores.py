#%%
from pathlib import Path
from typing import List

import pandas as pd
import typer
from psmpy_mod import PsmPy
from typer import Argument, Option


def iptw_from_psmpy(psm_data, endpoint):
    out = psm_data.copy()
    iptw_scores = []
    counter = {"pos": 0, "neg": 0}
    for i, row in psm_data.iterrows():
        positive = row[endpoint] == 1
        if positive:
            iptw_scores.append(1 / row["propensity_score"])
            counter["pos"] += 1
        else:
            iptw_scores.append(1 / (1 - row["propensity_score"]))
            counter["neg"] += 1

    print(counter)
    out["iptw_score"] = iptw_scores
    return out


def main(
    output_dir: Path = Argument(...),
    train_data: Path = Argument(...),
    other_datasets: List[Path] = Argument(...),
    endpoints: List[str] = Option(..., "-e", "--endpoints"),
    variables: List[str] = Option(["age", "Male"], "-v", "--variables"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    columns_of_interest = endpoints + variables

    train_df = pd.read_csv(str(train_data), index_col="patient_id").loc[
        :, columns_of_interest
    ]
    other_dfs = pd.concat(
        [
            pd.read_csv(str(file), index_col="patient_id").loc[:, columns_of_interest]
            for file in other_datasets
        ]
    )
    whole_df = pd.concat([train_df, other_dfs])

    for endpoint in endpoints:
        to_ignore = [ep for ep in endpoints if ep != endpoint]
        psm_train = PsmPy(
            train_df.reset_index(),
            treatment=endpoint,
            indx="patient_id",
            exclude=to_ignore,
        )
        psm_train.logistic_ps(balance=False)
        other_scores = psm_train.apply_fitted_model(other_dfs.reset_index())
        combined = pd.concat([psm_train.predicted_data, other_scores])
        combined_scores = iptw_from_psmpy(combined, endpoint).loc[
            :, ["patient_id", "propensity_logit", "propensity_score", "iptw_score"]
        ]
        ps_subset = whole_df.drop(to_ignore, axis=1)

        final_scores = pd.merge(ps_subset, combined_scores, on="patient_id", how="left")
        final_scores.to_csv(output_dir / f"iptw-scores_{endpoint}.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
