"""
Named sensory inputs and behavioural outputs of the fly brain, defined by
FlyWire cell-type annotations (Schlegel et al. 2024). The console, the game and
the experiments all use these groups, so a sense means the same neurons
everywhere.

Senses come in left/right pairs (the annotation's `side` column). Outputs are
motor or descending neurons whose role is known from the literature.
"""
JO_CE = ["JO-CM", "JO-CL", "JO-CA1", "JO-CA2", "JO-EV1", "JO-EV2", "JO-EV3",
         "JO-EV4", "JO-EV5", "JO-EV6", "JO-ED1", "JO-ED2_a", "JO-ED2_b", "JO-ED2_c"]
JO_AB = ["JO-A1", "JO-A2", "JO-A3", "JO-A4", "JO-A5", "JO-B1_a", "JO-B1_b",
         "JO-B1_c", "JO-B2", "JO-B3", "JO-B4_a", "JO-B4_b"]
BITTER = ["LB1a,LB1d", "LB1b", "LB1c", "LB1e"]

SENSES = [
    ("sugar", "taste", "Sugar  (LB3 GRNs)", {"cell_type": "LB3"}),
    ("bitter", "taste", "Bitter  (LB1 GRNs)", {"cell_type": BITTER}),
    ("salt", "taste", "Low salt  (LB2/LB4)", {"cell_sub_class": "low-salt"}),
    ("loom", "vision", "Looming  (LC4 + LPLC2)", {"cell_type": ["LC4", "LPLC2"]}),
    ("wind", "antenna", "Wind / gravity  (JO-C/E)", {"cell_type": JO_CE}),
    ("sound", "antenna", "Sound  (JO-A/B)", {"cell_type": JO_AB}),
    ("vinegar", "smell", "Vinegar  (ORN DM1)", {"cell_type": "ORN_DM1"}),
    ("geosmin", "smell", "Geosmin  (ORN DA2)", {"cell_type": "ORN_DA2"}),
    ("co2", "smell", "CO2  (ORN V)", {"cell_type": "ORN_V"}),
]

MODALITIES = ["taste", "vision", "antenna", "smell"]

OUTPUTS = [
    ("mn9", "MN9", {"cell_type": "CB0701"}, None, "proboscis extension"),
    ("gf_L", "Giant fiber L", {"cell_type": "DNp01"}, "left", "escape jump"),
    ("gf_R", "Giant fiber R", {"cell_type": "DNp01"}, "right", "escape jump"),
    ("dna01_L", "DNa01 L", {"cell_type": "DNa01"}, "left", "steering"),
    ("dna01_R", "DNa01 R", {"cell_type": "DNa01"}, "right", "steering"),
    ("dna02_L", "DNa02 L", {"cell_type": "DNa02"}, "left", "steering"),
    ("dna02_R", "DNa02 R", {"cell_type": "DNa02"}, "right", "steering"),
    ("mdn", "MDN", {"cell_type": "MDN"}, None, "backward walking"),
]

SIDES = ("left", "right")


class Circuits:
    """Resolves the named groups to neuron indices of a FlyBrain."""

    def __init__(self, brain):
        self.brain = brain
        self.sense = {}
        for key, _, _, match in SENSES:
            for side in SIDES:
                self.sense[key, side] = brain.neurons(side=side, **match)
        self.out = {key: brain.neurons(side=side, **match)
                    for key, _, match, side, _ in OUTPUTS}
        self.out_label = {key: label for key, label, *_ in OUTPUTS}

    def drive(self, rates):
        """{(sense key, side): Hz} -> input list for FlyBrain.set_input."""
        return [(self.sense[k], hz) for k, hz in rates.items() if hz > 0]

    def rates(self, spike_delta, ms):
        """Mean rate (Hz) of each output group from a spike-count difference."""
        return {k: float(spike_delta[i].mean()) / (ms * 1e-3) if len(i) else 0.0
                for k, i in self.out.items()}

    def fired(self, fired_idx):
        """Output keys with at least one spike among `fired_idx`."""
        s = set(fired_idx.tolist()) if len(fired_idx) else set()
        return [k for k, i in self.out.items() if any(j in s for j in i)]


if __name__ == "__main__":
    from fly_brain import FlyBrain
    c = Circuits(FlyBrain())
    for (key, side), idx in c.sense.items():
        print(f"{key:8s} {side:5s} {len(idx):4d} neurons")
    for key, idx in c.out.items():
        print(f"{key:8s} {len(idx):4d} neurons")
    assert all(len(v) for v in c.out.values()), "an output group is empty"
    assert all(len(v) for v in c.sense.values()), "a sense group is empty"
