import numpy as np
import pandas as pd

from itertools import combinations

from cmonge.metrics import average_r2, compute_scalar_mmd, wasserstein_distance


def score_transports_and_targets_combinations(all_expr, all_meta):
    target_mmds = []
    target_wds = []
    target_R2s = []

    transport_mmds = []
    transport_wds = []
    transport_R2s = []

    tt_mmds = []
    tt_wds = []
    tt_R2s = []
    target_transport_cars = []

    for CAR1, CAR2 in list(combinations(all_meta["condition"].unique(), 2)):
        target1 = np.array(
            all_expr[(all_meta["condition"] == CAR1) & (all_meta["dtype"] == "target")]
        )
        target2 = np.array(
            all_expr[(all_meta["condition"] == CAR2) & (all_meta["dtype"] == "target")]
        )

        transport1 = np.array(
            all_expr[
                (all_meta["condition"] == CAR1) & (all_meta["dtype"] == "transport")
            ]
        )
        transport2 = np.array(
            all_expr[
                (all_meta["condition"] == CAR2) & (all_meta["dtype"] == "transport")
            ]
        )

        target_mmd = compute_scalar_mmd(target1, target2)
        target_wd = wasserstein_distance(target1, target2)
        target_R2 = average_r2(target1, target2)

        transport_mmd = compute_scalar_mmd(transport1, transport2)
        transport_wd = wasserstein_distance(transport1, transport2)
        transport_R2 = average_r2(transport1, transport2)

        target_mmds.append(target_mmd)
        target_wds.append(target_wd)
        target_R2s.append(target_R2)
        transport_mmds.append(transport_mmd)
        transport_wds.append(transport_wd)
        transport_R2s.append(transport_R2)

        if CAR1 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target1, transport1)
            tt_wd = wasserstein_distance(target1, transport1)
            tt_R2 = average_r2(target1, transport1)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_R2s.append(tt_R2)
            target_transport_cars.append(CAR1)

        if CAR2 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target2, transport2)
            tt_wd = wasserstein_distance(target2, transport2)
            tt_R2 = average_r2(target2, transport2)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_R2s.append(tt_R2)
            target_transport_cars.append(CAR2)

    scores = pd.DataFrame(
        {
            "scores": (
                target_mmds
                + transport_mmds
                + tt_mmds
                + target_R2s
                + transport_R2s
                + tt_R2s
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
