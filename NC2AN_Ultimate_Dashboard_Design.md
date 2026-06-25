# NC²AN Ultimate Dashboard Design

## 1. Streamlit Page Architecture
**Theme**: ISRO Mission Control (Dark Mode, high contrast, sans-serif or monospace fonts).
**Layout**: Wide mode (`layout="wide"`).
**Structure**: A responsive grid dividing the screen into three primary columns:
- **Left Column**: Live telemetry and input streams.
- **Center Column**: Explainability and core forecast engines (The heart of NC²AN).
- **Right Column**: Operational metrics, diagnostics, and defense controls.

## 2. Component Hierarchy & Panel Design

### Panel A: Solar Activity Timeline
- **Purpose**: Macro-view of solar flux over the last 24 hours.
- **Inputs**: Aggregated 1-minute resolution flux values.
- **Visual Design**: Sleek, glowing line chart with anomalous regions highlighted.
- **Graph Type**: Plotly Interactive Line Chart.
- **Data Source**: Historical dataset stream (SOLEXS baseline).
- **Judge Value**: Anchors the user in the "big picture" context before diving into micro-seconds.
- **Implementation Complexity**: Low.

### Panel B: HEL1OS Hard X-Ray Stream
- **Purpose**: Real-time display of the exact 60-second input window for the Query.
- **Inputs**: `[60, 1]` HEL1OS tensor.
- **Visual Design**: Cyan traces, rapid updates, high frequency appearance.
- **Graph Type**: Real-time updating line plot (Streamlit `line_chart` or Altair).
- **Data Source**: Notebook 1 HEL1OS buffer.
- **Judge Value**: Demonstrates the pipeline gracefully handling high-frequency raw instrument data.
- **Implementation Complexity**: Low.

### Panel C: SOLEXS Soft X-Ray Stream
- **Purpose**: Real-time display of the exact 60-second input window for the Key/Value.
- **Inputs**: `[60, 4]` SOLEXS tensor.
- **Visual Design**: Magenta/Orange traces.
- **Graph Type**: Multi-line plot synchronized to Panel B's x-axis.
- **Data Source**: Notebook 1 SOLEXS buffer.
- **Judge Value**: Shows the delayed plasma heating response physically linked to Panel B.
- **Implementation Complexity**: Low.

### Panel D: Flare Probability Engine
- **Purpose**: The primary operational output determining space weather risk.
- **Inputs**: Model forecast outputs.
- **Visual Design**: Large, bold digital readouts with dynamic color coding (Green -> Yellow -> Red) and warning klaxons if the TSS-calibrated threshold is exceeded.
- **Graph Type**: Circular gauge charts or KPI metric blocks.
- **Data Source**: Notebook 2 Inference Output (1h, 6h, 12h, 24h).
- **Judge Value**: The ultimate "so what?" of the mission.
- **Implementation Complexity**: Low.

### Panel E: Cross-Attention CTP Heatmap
- **Purpose**: Explain *why* the model made its prediction by tracing the Neupert causal delay.
- **Inputs**: `[60, 60]` Attention Matrix.
- **Visual Design**: High contrast 'inferno' colormap with a dashed cyan diagonal line representing zero-delay physical transfer.
- **Graph Type**: Seaborn or Plotly Heatmap.
- **Data Source**: Notebook 2 Inference Engine internal weights.
- **Judge Value**: Destroys the "Black Box" criticism. A major winning feature for scientific judges valuing thermodynamics.
- **Implementation Complexity**: Medium (requires careful axis alignment and normalization).

### Panel F: Neupert Deviation Index (NDI) Monitor
- **Purpose**: Flag events that break standard solar physics for further manual review.
- **Inputs**: Calculated NDI stream from Notebook 2's PINN loss function.
- **Visual Design**: Anomaly detection line chart with a fixed red critical threshold line. Spikes are distinctly annotated.
- **Graph Type**: Line chart with filled area beneath spikes.
- **Data Source**: Notebook 2 Custom Physics Loss Function output.
- **Judge Value**: Proves the model operates as a "scientific discovery engine", not just a blind statistical classifier.
- **Implementation Complexity**: Medium.

### Panel G: Forecast Center (1h, 6h, 12h, 24h)
- **Purpose**: Detailed probabilistic breakdown of upcoming windows.
- **Inputs**: MC Dropout mean and std dev inference outputs.
- **Visual Design**: Bar charts with error bars (1-sigma uncertainty bands).
- **Graph Type**: Bar or point chart with error-bars.
- **Data Source**: Notebook 2 MC Dropout inference.
- **Judge Value**: Evidences understanding of operational uncertainty; deterministic point forecasts are not viable in real space weather operations.
- **Implementation Complexity**: Medium.

### Panel H: Judge Explainability Mode
- **Purpose**: An interactive QA terminal designed exclusively for the competition defense/pitch.
- **Inputs**: Interactive buttons/prompts ("Why Cross Attention?", "Explain NDI", "How do you handle class imbalance?").
- **Visual Design**: A collapsible terminal/console-style text box with a typewriter effect.
- **Graph Type**: Interactive markdown text blocks.
- **Data Source**: Pre-written defense scripts (extracted from Notebook 3).
- **Judge Value**: Controls the narrative during the defense by pre-empting the judges' hardest questions.
- **Implementation Complexity**: Low.

### Panel I: Mission Health & GTI Status
- **Purpose**: System diagnostics and pipeline health metrics.
- **Inputs**: Internal Data Engineering flags (e.g., CUSUM Sentinel status).
- **Visual Design**: Minimalist glowing LED-style status indicators (Green/Red).
- **Graph Type**: Boolean indicator lights or text badges.
- **Data Source**: Notebook 1 Pipeline monitoring parameters.
- **Judge Value**: Instills confidence regarding production-readiness and fault tolerance.
- **Implementation Complexity**: Low.

## 3. User Flow & Navigation Design
1. **Idle State**: Panels B & C update quietly reflecting quiet sun. Panel D shows < 5% probabilities. The CUSUM indicator in Panel I is green ("Monitoring").
2. **Event Trigger**: A flare pulse begins. CUSUM Sentinel turns red. The dashboard visually escalates.
3. **Inference State**: Model engages. Panel D Probabilities spike across the horizons.
4. **Explainability Review**: The operator consults Panel E (CTP Heatmap) to verify a physical transfer delay is present, and Panel F (NDI) to ensure the prediction obeys Neupert physics or flags an anomaly.
5. **Action**: Relying on Panel G (Forecast Center), the operator issues official warnings with mathematically backed confidence intervals.

## 4. Demo Sequence for Judges (Winning Presentation Strategy)
1. **The Hook (0:00 - 1:00)**: Load the live Streamlit dashboard. Do not show code yet. Let judges observe the raw data flowing and the probability dials turning over.
2. **The Black Box Killer (1:00 - 2:30)**: Trigger a mock flare sequence. Immediately direct their attention to the CTP Heatmap (Panel E). Explain that NC²AN causally mapped HEL1OS to SOLEXS. Point explicitly to the cyan diagonal line.
3. **The Physics Engine (2:30 - 3:30)**: Explain the NDI (Panel F). Emphasize that while standard LSTMs don't know physics, NC²AN explicitly penalizes predictions that violate $dS/dt \propto H$.
4. **The Mic Drop (3:30 - 4:00)**: Open the "Judge Explainability Mode" (Panel H) and demonstrate how the dashboard itself directly answers the deepest technical questions about class imbalance (using TSS/HSS metrics) and thermodynamic constraints.

## 5. Summary
This design empowers a teammate to construct a fully robust frontend UI decoupled from the heavy backend AI modeling. The modular Streamlit approach combined with strict UI Panel definitions makes parallel development exceptionally smooth.
