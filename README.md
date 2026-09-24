# Fly Brain Rig & Fly Swat

The whole adult fruit-fly brain, running live: all 138,639 neurons and
54.5 million synapses of the [FlyWire](https://flywire.ai) connectome (v783),
simulated as leaky integrate-and-fire neurons and stepped in real time.

Two programs use the same brain:

- **Fly Brain Rig** (`fly_console.py`): a lab console. Switch senses on and
  off (taste, looming, antenna, smell), silence any cell type, and watch
  activity spread through the brain and reach the motor and descending neurons.
- **Fly Swat** (`fly_swat.py`): try to swat a fly. The swatter's image
  expanding on the fly's eye drives its looming detectors. Whether the fly
  escapes depends on whether its giant fiber spikes in time, and that is
  decided by the connectome.

![Fly Brain Rig](docs/console.png)
![Fly Swat](docs/swat.png)

## Quick start

```bash
git clone --recursive https://github.com/snuri00/flywire-connectome-swat.git
cd flywire-connectome-swat
pip install -r requirements.txt
python3 fly_console.py              # the lab console
python3 fly_swat.py                 # the game
python3 experiments.py              # stimulus -> response experiments
python3 fly_swat.py --headless 20   # measure escape rates, no window
```

The connectivity data come with the
[Shiu et al. model](https://github.com/philshiu/Drosophila_brain_model)
submodule. The first run downloads the FlyWire cell-type annotations
(31 MB) into `data/`. Loading the brain takes about 2 s and 1.5 GB of RAM.

## Fly Brain Rig

| Action | Effect |
|---|---|
| Click a sense's `L` / `R` chip | switch that side's sensory neurons on or off (Poisson input) |
| Click the sense's name | switch both sides |
| Mouse wheel over a sense | change its rate (25–400 Hz) |
| `/` then a cell type, `Enter` | stimulate every neuron of that type (e.g. `DNp01`, `LC6`) |
| `/`, `Tab`, cell type, `Enter` | silence that type (all its outgoing synapses set to zero) |
| Hover over the brain | name, class and rate of the nearest active neuron |
| `SPACE` / `1`–`4` | pause / model speed 0.1×, 0.25×, 0.5×, 1× |
| `C` / `R` / `E` | all senses off / reset brain state / export CSV to `recordings/` |

Panels:

- **Brain**: every neuron at its soma, frontal view. A neuron lights up in
  its class colour when it spikes and fades over ~150 ms, like a
  calcium-imaging movie. Rings mark the output neurons.
- **Output neurons**: live rate of the proboscis motor neuron MN9, the giant
  fibers, the steering neurons DNa01/DNa02 and the "moonwalker" MDN.
- **Most active cell types**: the pathway the signal is taking right now.
- **Stimulus → first output spike**: latency from switching a sense on to the
  first spike of each output. Outputs that were already firing are not counted.

## Fly Swat

| Input | Action |
|---|---|
| Mouse | aim the swatter (it hovers 14 cm above the table) |
| Left click | swing |
| Right click / Shift + right click | drop sugar water / bitter water |
| `1` `2` `3` | slow (220 ms), normal (130 ms), fast (80 ms) swing |
| `S` | slow motion during swings |
| `G` | cut the giant fibers |
| `N`, `SPACE`, `ESC` | new fly, pause, quit |

After every swing the **Last swing** panel shows a millisecond timeline: when
the eye started to report the looming swatter, when each giant fiber spiked,
when the fly took off, and when the swatter hit the table.

### What the connectome decides, and what is scripted

| Connectome (spiking model) | Scripted |
|---|---|
| **When the fly escapes**: looming drives LC4 and LPLC2 of the eye that sees the swatter; the fly takes off 6 ms after either giant fiber (DNp01) spikes | the walking path: a random walk, drawn to sugar drops within 7 cm |
| **Whether the fly feeds**: a drop under the fly drives the sugar (LB3) or bitter (LB1) taste neurons; the fly stops and drinks while MN9 fires | the direction of the jump (away from the swatter) and the flight path |
| **Bitter stops feeding**: bitter input silences MN9 | the smell of sugar is fed to the vinegar receptor neurons (ORN DM1) for display only |

The step from an expanding image to LC4/LPLC2 firing happens in the retina
and optic lobe, which the model does not simulate. It is set by hand: no
response below 3 rad/s of angular expansion, then 12 Hz per rad/s, up to 300 Hz.

## How it works

### Brain (`fly_brain.py`)

The neuron model, its constants and the synapse rule are those of
Shiu et al. 2024, *Nature* 634, 210–219:

    dv/dt = (v_0 - v + g) / t_mbr        v_0 = -52 mV, t_mbr = 20 ms
    dg/dt = -g / tau                      tau = 5 ms

A spike (v > -45 mV) resets v and g and makes the neuron refractory for
2.2 ms. After 1.8 ms each spike adds 0.275 mV × (signed synapse count) to the
g of every postsynaptic neuron. The sign comes from the predicted
neurotransmitter: acetylcholine excites, GABA and glutamate inhibit.

The paper runs whole trials in Brian2. Here the same model is written as
two numba loops over a sparse (CSR) weight matrix, so it can be stepped
0.1 ms at a time inside a game loop. It runs at about 0.9× real time on one
CPU core.

### Validation against the published model

Fed the same deterministic input as the original Brian2 code, the
reimplementation produces the **same 2,798 spikes, in the same neurons, at
the same time steps**. With Poisson input (sugar neurons at 100 Hz, 16 trials
each), the per-neuron firing rates correlate at r = 0.9994.

Getting there required one detail that is not in the paper: in Brian2,
synaptic input that reaches a neuron while it is refractory is **lost**
(the variables are marked "unless refractory"). A first version that kept
this input produced 18 % more spikes, and up to twice the rate in strongly
recurrent motor circuits.

## Measured behaviour

**Console experiments** (`experiments.py`; 5 trials × 1 s, inputs at 150 Hz)

| Stimulus | Result |
|---|---|
| none | no neuron fires; the model has no spontaneous activity |
| sugar neurons (LB3, left) | MN9 89 Hz: proboscis extension |
| sugar + bitter | MN9 7.7 Hz: bitter suppresses feeding |
| bitter alone | MN9 0 Hz |
| looming, left eye (LC4 + LPLC2) | giant fiber left 168 Hz, right 109 Hz; DNa01 right 41 Hz |
| looming, both eyes | both giant fibers 176–192 Hz |
| looming, both eyes, LC4 silenced | giant fibers still 150–170 Hz: LPLC2 alone is enough |
| antennal JO-C/E, left | descending neurons DNbe001, DNp18, DNge091 …; no steering or MDN activity |

**Giant-fiber latency** to a looming ramp on the left eye (first spike, ms)

| Ramp to peak | 50 Hz peak | 100 Hz peak | 200 Hz peak |
|---|---|---|---|
| 50 ms | 25–40 | 20–25 | 15 |
| 150 ms | 30–65 | 25–55 | 20–35 |
| 400 ms | 85–140 | 70–95 | 30–65 |

While the fly drinks sugar, the same 150 ms / 100 Hz ramp needs 95–120 ms.
Feeding delaying escape is not programmed anywhere: it comes out of the
connectome.

**Fly Swat** (`fly_swat.py --headless 20`; swatter aimed at the fly with a 15 mm aim error)

| Condition | Hit | Giant fiber, first spike (median) |
|---|---|---|
| walking fly, normal swing (130 ms) | 11 / 20 | 48 ms |
| drinking fly, normal swing | 13 / 20 | 70 ms |
| walking fly, fast swing (80 ms) | 16 / 20 | 25 ms |
| walking fly, slow swing (220 ms) | 2 / 20 | 102 ms |
| giant fibers cut, normal swing | 20 / 20 | 48 ms (spikes, but cannot jump) |

A drinking fly's giant fiber fires about 20 ms later, and the fly is hit more
often. A slow swing is seen coming; a fast one leaves too little time.

## Limitations

- Only the brain is modelled. The ventral nerve cord (which turns descending
  commands into leg and wing movements) is not, so walking, the jump direction
  and flight are scripted.
- The model has no spontaneous activity; without input the brain is silent.
- No walking command emerged from the stimuli tried so far: MDN stays nearly
  silent.
- The retina and early optic lobe are not simulated; looming enters at LC4/LPLC2
  through a hand-set mapping.
- No plasticity, neuromodulation or gap junctions; neuron parameters are the
  same for every cell.
- The `side` of each neuron is taken from the FlyWire annotations; whether it
  is the fly's left or the left of the image volume has not been checked.
- Real time is reached only just (≈0.9× on one core); the game shows the
  brain's clock.

## Files

| File | Purpose |
|---|---|
| `fly_brain.py` | connectome loader and integrate-and-fire network (numba); run it for a short demo |
| `fly_circuits.py` | named sensory inputs and motor/descending outputs (by cell type) |
| `brain_view.py` | live picture of all neurons at their soma positions |
| `fly_console.py` | Fly Brain Rig console (pygame) |
| `fly_sprite.py` | fly, swatter and drop sprites, rendered at start-up from shaded distance fields |
| `swat_world.py` | the swatting world: fly, swatter, drops, looming, brain loop |
| `fly_swat.py` | Fly Swat game (pygame) and headless benchmark |
| `experiments.py` | stimulus → response experiments from the command line |
| `Drosophila_brain_model/` | Shiu et al. model (git submodule): connectivity data, Brian2 reference |

## Credits

- Connectome: FlyWire Consortium, Dorkenwald et al., *Nature* 634, 124–138 (2024).
- Cell types and annotations: Schlegel et al., *Nature* 634, 139–152 (2024).
- Neurotransmitter predictions: Eckstein et al., *Cell* 187, 2574–2594 (2024).
- Whole-brain LIF model, parameters and data files: Shiu et al., *Nature* 634, 210–219 (2024).
- Looming detection and the giant fiber: von Reyn et al., *Nat. Neurosci.* 17, 962–970 (2014);
  Ache et al., *Curr. Biol.* 29, 1073–1081 (2019).
