# DS Team AI4Football

## Method

The solution estimates home possession (H), away possession (A), and dead ball (D) from player and ball tracking. A spatial graph encoder and bidirectional temporal GRU use positions, motion, team context, and missing-observation masks. Two fixed models with 6-second and 4-second context produce probabilities at 5 Hz, combined with weights 0.75 and 0.25. Fixed Viterbi decoding penalizes state changes, retains possession during missing ball observations, and uses pitch boundaries to refine dead-ball segments. Predictions are expanded to the original 25 Hz frame grid.

Weights and decoding parameters are included. Inference is offline and uses future context. Tracking gaps and identity errors can affect possession boundaries.

The models use public Bundesliga tracking from [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y) ([dataset](https://doi.org/10.6084/m9.figshare.28196177)).

## Run

Tested with Python 3.13 on CPU. From the repository root:

```bash
python -m pip install -r solution/requirements.txt
python -B solution/predict.py \
  --tracking data/tactical/BAR-ATH_tactical_atd.txt \
  --metadata data/tactical/BAR-ATH_tactical_atd.xml \
  --output "DS Team AI4Football.csv"
```
