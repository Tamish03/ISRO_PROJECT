# 🌞 NC²AN COMPLETE PREDICTION DNA
### (End-to-End Flow of Notebook 2)

---

# INPUT FROM NOTEBOOK 1

For every second inside a 60-minute window, satellites observe:

    • Hard X-Ray Flux (HEL1OS)
    • Soft X-Ray Flux (SoLEXS)

From these, Notebook 1 engineers five scientific features.

For each second:

    1. Hard Flux
    2. Soft Flux
    3. Hard / Soft Ratio
    4. dSoft / dt
    5. Total Energy

Example

Second 1

[
 Hard=15,
 Soft=28,
 H/S=0.54,
 dS/dt=1.2,
 Energy=42
]

Entire Batch

Shape

Batch × Time × Features

[32 × 60 × 5]

------------------------------------------------------------

# STEP 1 — TEMPORAL ENCODER
### Convert Scientific Features → Latent Embeddings

Raw scientific values cannot express complex flare patterns.

Therefore,

each

5-dimensional feature vector

is projected into

64-dimensional latent space.

Example

Before

Second 1

[15,28,0.54,1.2,42]

↓

Temporal Encoder

↓

After

[
-0.31,
 1.84,
 0.56,
...
 0.74
]

(64 dimensions)

Every second now becomes

64 learned scientific features.

Entire Tensor

Before

[32 × 60 × 5]

↓

After

[32 × 60 × 64]

These are called

LATENT EMBEDDINGS

------------------------------------------------------------

# STEP 2 — CROSS ATTENTION
### Learn Hard ↔ Soft X-Ray Relationships

Now every Hard X-Ray embedding asks:

    "Which Soft X-Ray observations help explain me?"

Example

Hard Second 2

[6,5,7]

Instead of only looking at

Soft Second 2

the model learns

Soft Second 1

20%

Soft Second 2

35%

Soft Second 3

45%

Weighted Soft Information

=

0.20×S1
+
0.35×S2
+
0.45×S3

↓

Hard embedding gets updated.

Example

Before

Second 1

[1,2,1]

↓

Cross Attention

↓

Second 1

[2,4,3]

Before

Second 2

[4,3,5]

↓

Cross Attention

↓

Second 2

[6,5,7]

Before

Second 3

[8,7,8]

↓

Cross Attention

↓

Second 3

[9,8,10]

These are now

Physics-aware latent embeddings.

IMPORTANT

The output shape DOES NOT change.

It remains

[32 × 60 × 64]

Only the information inside every embedding becomes richer.

------------------------------------------------------------

# STEP 3 — RECONSTRUCTION HEAD
### Predict Future Soft X-Ray Flux

Prediction now begins.

Every latent embedding predicts

the Soft X-Ray Flux

for its corresponding second.

Example

Second 1

Embedding

[2,4,3]

↓

Predicted Flux

45

Second 2

Embedding

[6,5,7]

↓

Predicted Flux

74

Second 3

Embedding

[9,8,10]

↓

Predicted Flux

120

Entire predicted sequence

[45,74,120]

Ground Truth

[47,72,118]

------------------------------------------------------------

# DATA LOSS

Compare

Prediction

↓

Ground Truth

Example

Prediction

45

74

120

Ground Truth

47

72

118

↓

Mean Squared Error

↓

Data Loss

This teaches the model

to reconstruct realistic

Soft X-Ray evolution.

------------------------------------------------------------

# PHYSICS LOSS (NEUPERT LOSS)

Now

the model checks whether

its prediction obeys

Solar Physics.

Compute

Derivative

Predicted Flux

45

↓

74

↓

120

↓

dSoft/dt

[29,46]

Ground Truth Hard Flux

[31,44]

Physics says

dSoft/dt

∝

Hard Flux

Since

29≈31

46≈44

↓

Physics Loss is SMALL.

If

Predicted Flux

[45,46,47]

↓

Derivative

[1,1]

Hard Flux

[31,44]

↓

Huge mismatch

↓

Physics Loss becomes LARGE.

Therefore

the reconstruction head has

TWO teachers.

Teacher 1

Data Loss

Teacher 2

Physics Loss

------------------------------------------------------------

# STEP 4 — TEMPORAL ATTENTION POOLING
### Find Which Seconds Matter Most

We still have

60 latent embeddings.

But

classification only needs

ONE representation.

Instead of averaging,

the model learns

which seconds are important.

Example

Second

1

↓

Weight

0.10

Second

2

↓

Weight

0.30

Second

3

↓

Weight

0.60

Weighted Sum

=

0.1×[2,4,3]

+

0.3×[6,5,7]

+

0.6×[9,8,10]

↓

Master Embedding

[7.4,6.7,8.4]

Shape

Before

[60 × 64]

↓

After

[64]

This vector summarizes

the ENTIRE solar event.

------------------------------------------------------------

# STEP 5 — SHARED TRUNK
### Build One Master Understanding

Master Embedding

[7.4,6.7,8.4]

↓

Dense

↓

LayerNorm

↓

Dropout

↓

Master Representation

[8.1,7.2,9.0]

The model now understands

the complete flare behaviour

instead of individual seconds.

------------------------------------------------------------

# STEP 6 — MULTI-HORIZON FORECAST HEADS

The SAME master representation

answers FOUR questions.

Head 1

Will flare occur

within 1 hour?

↓

Dense

↓

Sigmoid

↓

97.8%

--------------------------------

Head 2

Will flare occur

within 6 hours?

↓

90.0%

--------------------------------

Head 3

Will flare occur

within 12 hours?

↓

73.1%

--------------------------------

Head 4

Will flare occur

within 24 hours?

↓

40.1%

Notice

the model predicts

PROBABILITIES

not simply

YES

or

NO.

------------------------------------------------------------

# STEP 7 — FORECAST LOSS

Notebook 1 provides

Ground Truth Labels.

Example

1 Hour

1

6 Hours

1

12 Hours

1

24 Hours

0

Prediction

0.978

0.900

0.731

0.401

↓

Binary Cross Entropy

↓

Forecast Loss

------------------------------------------------------------

# STEP 8 — TOTAL LOSS

Now all objectives are combined.

Forecast Loss

+

Data Reconstruction Loss

+

λ × Physics Loss

↓

Total Loss

Example

Forecast

0.12

Data

0.05

Physics

0.02

λ

0.5

↓

Total Loss

=

0.12

+

0.05

+

0.5×0.02

=

0.18

------------------------------------------------------------

# STEP 9 — BACKPROPAGATION

This Total Loss is propagated

through the ENTIRE architecture.

Total Loss

↓

GradientTape

↓

Adam Optimizer

↓

Prediction Heads Updated

↓

Shared Trunk Updated

↓

Temporal Attention Updated

↓

Reconstruction Head Updated

↓

Feed Forward Updated

↓

Cross Attention Updated

↓

RoPE Updated

↓

Temporal Encoder Updated

Millions of windows later,

the network gradually learns

✔ Which Hard X-Ray patterns precede flares

✔ Which Soft X-Ray behaviour is important

✔ Which timestamps deserve attention

✔ Which latent representations best describe solar activity

✔ How to obey the Neupert Effect

✔ How to forecast future flare probabilities

------------------------------------------------------------

# 🧬 FINAL NC²AN DNA

Notebook 1
──────────────

Raw Satellite Data

↓

Feature Engineering

↓

Hard Flux
Soft Flux
H/S Ratio
dSoft/dt
Energy

↓

[32 × 60 × 5]

────────────────────────────────────────────

Notebook 2

↓

Temporal Encoder

↓

[32 × 60 × 64]

↓

Cross Attention

(Hard ↔ Soft Interaction)

↓

Physics-Aware Latent Embeddings

↓

├───────────────────────────────────────┐
│                                       │
│ Reconstruction Head                   │
│                                       │
│ Predict Soft X-Ray Sequence           │
│                                       │
│ ├──► Data Loss                        │
│ └──► Physics Loss (Neupert)           │
│                                       │
└───────────────────────────────────────┘

↓

Temporal Attention Pooling

↓

Master Embedding

↓

Shared Trunk

↓

Multi-Horizon Heads

├──► Flare within 1 Hour

├──► Flare within 6 Hours

├──► Flare within 12 Hours

└──► Flare within 24 Hours

↓

Forecast Loss

↓

Total Loss

Forecast
+
Data
+
λ × Physics

↓

Backpropagation

↓

Updated NC²AN

↓

Repeat for millions of windows

↓

FINAL MODEL

"Predicts solar flare probabilities while simultaneously learning
scientific representations that are constrained by the Neupert Effect,
making the model both data-driven and physics-informed."
