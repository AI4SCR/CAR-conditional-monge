from itertools import combinations

import numpy as np
import pandas as pd
from cmonge.metrics import average_r2, compute_scalar_mmd, wasserstein_distance


def score_transports_and_targets_combinations(all_expr, all_meta):
    target_mmds = []
    target_wds = []
    target_r2s = []

    transport_mmds = []
    transport_wds = []
    transport_r2s = []

    tt_mmds = []
    tt_wds = []
    tt_r2s = []
    target_transport_cars = []

    for car1, car2 in list(combinations(all_meta["condition"].unique(), 2)):
        target1 = np.array(
            all_expr[(all_meta["condition"] == car1) & (all_meta["dtype"] == "target")]
        )
        target2 = np.array(
            all_expr[(all_meta["condition"] == car2) & (all_meta["dtype"] == "target")]
        )

        transport1 = np.array(
            all_expr[
                (all_meta["condition"] == car1) & (all_meta["dtype"] == "transport")
            ]
        )
        transport2 = np.array(
            all_expr[
                (all_meta["condition"] == car2) & (all_meta["dtype"] == "transport")
            ]
        )

        target_mmd = compute_scalar_mmd(target1, target2)
        target_wd = wasserstein_distance(target1, target2)
        target_r2 = average_r2(target1, target2)

        transport_mmd = compute_scalar_mmd(transport1, transport2)
        transport_wd = wasserstein_distance(transport1, transport2)
        transport_r2 = average_r2(transport1, transport2)

        target_mmds.append(target_mmd)
        target_wds.append(target_wd)
        target_r2s.append(target_r2)
        transport_mmds.append(transport_mmd)
        transport_wds.append(transport_wd)
        transport_r2s.append(transport_r2)

        if car1 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target1, transport1)
            tt_wd = wasserstein_distance(target1, transport1)
            tt_r2 = average_r2(target1, transport1)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_r2s.append(tt_r2)
            target_transport_cars.append(car1)

        if car2 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target2, transport2)
            tt_wd = wasserstein_distance(target2, transport2)
            tt_r2 = average_r2(target2, transport2)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_r2s.append(tt_r2)
            target_transport_cars.append(car2)

    scores = pd.DataFrame(
        {
            "scores": (
                target_mmds
                + transport_mmds
                + tt_mmds
                + target_r2s
                + transport_r2s
                + tt_r2s
                + target_wds
                + transport_wds
                + tt_wds
            ),
            "score": ["MMD"] * (2 * len(target_mmds) + len(tt_mmds))
            + ["R2"] * (2 * len(target_mmds) + len(tt_mmds))
            + ["WD"] * (2 * len(target_mmds) + len(tt_mmds)),
            "target_transport": (
                ["between targets"] * len(target_mmds)
                + ["between transports"] * len(transport_mmds)
                + ["between target transport"] * len(tt_mmds)
            )
            * 3,
        }
    )
    scores["scores"] = scores["scores"].astype(float)

    return scores
