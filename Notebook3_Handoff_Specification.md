# NC²AN Notebook 3: Handoff Specification

## 1. Overview
This specification provides the strict Interface Contracts between Notebooks 1 & 2 and Notebook 3. A teammate can build the complete Dashboard (Notebook 3) independently by mocking exactly these tensor shapes, datatypes, and physical definitions without needing to read the upstream source code.

## 2. Input Tensor Specifications (From Notebook 1)

**HEL1OS (Hard X-Rays)**
- **Physical Meaning**: Non-thermal electron acceleration.
- **Bandpass**: Broadband 1.8–90 keV.
- **Cadence**: 1 second (1 Hz).
- **Window Size**: 60 timesteps (1 minute).
- **Shape**: `[Batch_Size, 60, 1]`
- **Datatype**: `tf.float32`

**SOLEXS (Soft X-Rays)**
- **Physical Meaning**: Coronal plasma heating.
- **Unit**: Soft X-Ray counts.
- **Cadence**: 1 second (1 Hz).
- **Window Size**: 60 timesteps (1 minute).
- **Shape**: `[Batch_Size, 60, 4]` (1 channel for counts + 3 engineered HOPE features).
- **Datatype**: `tf.float32`

**Synchronization Rules**
- **Alignment Method**: `pandas.merge_asof` (done upstream in NB1).
- **Tolerance**: 0.5 seconds.
- **Missing Data**: Forward-filled upstream. The Dashboard should assume dense, valid tensors.

## 3. Model Output Specifications (From Notebook 2)

The loaded Keras model accepts the tuple `(hel1os_tensor, solexs_tensor)` and returns `(forecasts, attention_matrix)`.

**1. Flare Probabilities (Forecasts)**
- **Horizons**: 1h, 6h, 12h, 24h.
- **Activation**: Sigmoid probability $\in [0, 1]$.
- **Shape**: Tuple of 4 tensors, each `[Batch_Size]`.

**2. Cross-Attention Matrix (CTP)**
- **Physical Meaning**: Causal Transfer Proxy indicating which Hard X-Ray pulse caused which Soft X-Ray heating (Neupert Effect visualizer).
- **Shape**: `[Batch_Size, 60, 60]` (Query=HEL1OS [60], Key/Value=SOLEXS [60]).

**3. Neupert Deviation Index (NDI) & Loss**
- **Physical Meaning**: The ratio of the physics violation to the data error. Determines if a flare violates $dS/dt \propto H$.
- **Shape**: Scalar per window or calculated per sequence/batch `[Batch_Size]`.

## 4. Interface Contracts Between Notebooks

To build Notebook 3 immediately, implement the following mock classes in the Dashboard:

```python
import tensorflow as tf

def get_live_data_batch(batch_size=1):
    """
    Simulates Notebook 1 Data Engineering output.
    Returns:
        hel1os_batch: tf.Tensor of shape (batch_size, 60, 1)
        solexs_batch: tf.Tensor of shape (batch_size, 60, 4)
    """
    hel1os = tf.random.normal((batch_size, 60, 1))
    solexs = tf.random.normal((batch_size, 60, 4))
    return hel1os, solexs

class MockNC2AN:
    """
    Simulates Notebook 2 Architecture inference output.
    """
    def predict(self, hel1os, solexs):
        """
        Takes inputs from NB1 and returns mock NB2 predictions.
        """
        batch_size = tf.shape(hel1os)[0]
        
        # Flare probabilities for 1h, 6h, 12h, 24h
        preds = (
            tf.random.uniform((batch_size,)),
            tf.random.uniform((batch_size,)),
            tf.random.uniform((batch_size,)),
            tf.random.uniform((batch_size,))
        )
        
        # Cross-Attention / CTP
        ctp = tf.random.uniform((batch_size, 60, 60))
        
        # NDI array for operational monitoring
        ndi = tf.random.exponential(scale=0.5, size=(batch_size,))
        
        return preds, ctp, ndi
```

Integrate Notebook 3 with these mock structures. Final integration only requires swapping the data loader and the `MockNC2AN` class with standard TensorFlow `load_model` calls.
