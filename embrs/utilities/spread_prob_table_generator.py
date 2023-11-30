"""Script demonstrating how the fuel spread probabilities were generated
"""

import numpy as np

original_probs = {
    "sh": {"sh": 0.375, "gr": 0.35, "fpc": 0.4, "nfp": 0.375},
    "gr": {"sh": 0.475, "gr": 0.475, "fpc": 0.475, "nfp": 0.475},
    "fpc": {"sh": 0.325, "gr": 0.25, "fpc": 0.35, "nfp": 0.35},
    "nfp": {"sh": 0.1, "gr": 0.075, "fpc": 0.275, "nfp": 0.075}
}

# deltas
F1_D = -0.025
F2_D = 0
F3_D = 0.025
F4_D = 0.025
F5_D = -0.0125
F6_D = 0.0125
F7_D = -0.025
F8_D = -0.025
F9_D = 0.025
F10_D = -0.025
F11_D = 0
F12_D = 0
F13_D = +0.025

delta_mapping = {
    "f1": F1_D, "f2": F2_D, "f3": F3_D, "f4": F4_D, "f5": F5_D, "f6": F6_D, "f7": F7_D,
    "f8": F8_D, "f9": F9_D, "f10": F10_D, "f11": F11_D, "f12": F12_D, "f13": F13_D
}

category_map = {
    "f1": "gr", "f2": "gr", "f3": "gr", "f4": "sh", "f5": "sh", "f6": "sh", "f7": "sh",
    "f8": "nfp", "f9": "nfp", "f10": "fpc", "f11": "nfp", "f12": "fpc", "f13": "fpc"
}

nom_spread_prob_table = {}

for i, fuel1 in enumerate(category_map.keys()):
    print(f"{fuel1} probs:")
    fuel1_category = category_map[fuel1]
    nom_spread_prob_table[i+1] = {}

    for j, fuel2 in enumerate(category_map.keys()):
        fuel2_category = category_map[fuel2]
        original_prob = original_probs[fuel1_category][fuel2_category]
        prob = original_prob + delta_mapping[fuel1] + delta_mapping[fuel2]

        print(f"{j+1}: {np.round(prob, 3)}")

        nom_spread_prob_table[i+1][j+1] = np.round(prob,4)

    print("")
    print("")

print(f"final table: {nom_spread_prob_table}")
