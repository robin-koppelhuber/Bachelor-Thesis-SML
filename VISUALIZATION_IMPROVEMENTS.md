# Visualization Improvements Summary

## Overview
This document summarizes the improvements made to the multi-objective optimization benchmark visualizations for the Bachelor Thesis at ETH Zurich.

## Changes Made

### 1. **Commented Out Elements** ✅

#### Pareto Frontier Plots
- **Hypervolume shading**: Removed the green filled area that cluttered the plots
- **Nadir point marker**: Removed the purple square marker that wasn't meaningful

**File**: `src/visualization/pareto.py` (lines 89-109, 162-173)

### 2. **Improved Labels Across All Plots** ✅

#### New Utility Functions
Added formatting functions for consistent labeling:
- `format_task_name()`: Converts `ag_news` → `AG News`, `imdb` → `IMDB`, etc.
- `format_metric_name()`: Converts `f1_macro` → `F1 Macro`, etc.

**File**: `src/visualization/plots.py` (lines 18-52)

#### Updated Plots:
1. **Pareto Frontier**:
   - Task names properly formatted on axes
   - Single-task optima labels: "Fine-tuned: AG News" instead of "Ag News Optimal"
   - Better color scheme for multiple fine-tuned models
   - **Smart point labeling**: Only labels key points (closest to utopia, best on each axis)
   - Displays count of unlabeled Pareto points to avoid clutter

2. **Performance Heatmap**:
   - Formatted task names
   - Bold axis labels
   - Proper metric name formatting

3. **Radar Charts**:
   - Formatted task names
   - Better titles

4. **Task Interference Matrix**:
   - Formatted task names
   - Bold axis labels

5. **Preference Alignment**:
   - **IMPROVED**: Clarified normalization method
   - Y-axis: "Relative Weight" instead of "Weight"
   - Legend: "Achieved (÷ sum)" to show normalization
   - Added explanation text box: "Achieved normalized by sum of all task scores"
   - Better axis labels distinguishing preference from performance

### 3. **New Visualizations** ✅

#### Parallel Coordinates Plot
**Purpose**: Compare all preference vectors across all tasks simultaneously

**Features**:
- Each line represents one preference vector
- Color-coded by preference type:
  - 🟢 Green: Equal preference (all weights equal)
  - 🔴 Red: Extreme preference (one weight ≥ 0.7)
  - 🔵 Blue: Balanced preference
- Shows fine-tuned single-task performance as gold squares
- Easy to spot which preference vectors excel on which tasks

**File**: `src/visualization/plots.py` (lines 534-637)

#### Distance to Utopia Analysis
**Purpose**: Show which preference vectors achieve solutions closest to the ideal (utopia) point

**Features**:
- Horizontal bar chart sorted by distance (lower is better)
- Color-coded by preference type (same scheme as parallel coordinates)
- Shows exact distance values
- Displays the utopia point coordinates
- Helps identify which preference weighting strategies work best

**File**: `src/visualization/plots.py` (lines 640-762)

#### Performance Recovery Analysis
**Purpose**: Test hypothesis that multi-task model should achieve at least `preference_weight × single_task_performance` on each task

**Features**:
- Two-panel visualization showing both absolute values and percentages
- **Left Panel (Absolute Values)**:
  - Expected performance: `preference_weight × single_task_performance`
  - Achieved performance: actual multi-task model results
  - Side-by-side bars for direct comparison
- **Right Panel (Recovery Percentage)**:
  - Recovery rate: `(achieved / expected) × 100`
  - Dynamic color coding:
    - 🟢 Green (≥100%): Exceeds expectation (positive transfer/synergy)
    - 🟠 Orange (80-100%): Close to expectation (minimal interference)
    - 🔴 Red (<80%): Underperforming (negative transfer/interference)
  - 100% baseline for easy comparison
- Groups by preference vector type (equal/extreme/balanced)
- Includes explanation text boxes on both panels

**File**: `src/visualization/plots.py` (lines 770-937)

### 4. **Comprehensive Results Export** ✅

#### Export Function
Creates multiple output files with all metrics from all evaluated models:

**Outputs**:
1. **comprehensive_results.json**: Full dataset including:
   - All metrics for each (preference_vector, task) combination
   - Results from single-task fine-tuned models
   - Summary statistics
   - Best preference vector per task per metric
   - Distance to utopia for each preference vector

2. **comprehensive_results.csv**: Tabular format of all results

3. **summary_statistics.csv**: Average performance per task

4. **utopia_distances.csv**: Distance calculations for all preference vectors

**File**: `src/visualization/generator.py` (lines 448-646)

### 5. **Integration** ✅

The new visualizations and export are automatically generated during benchmark runs.

**File**: `src/benchmarks/poc/run.py` (lines 788-811)

## Color Coding Philosophy

### All Metrics Are "Higher is Better"
All evaluation metrics (accuracy, F1, precision, recall, MCC, Cohen's Kappa) follow the convention that **higher values are better**.

### Color Schemes:
- **Heatmaps**: `RdYlGn` (Red-Yellow-Green) where green = high (good), red = low (bad)
- **Task Interference**: `RdBu_r` where red = positive correlation (synergy), blue = negative (interference)
- **Preference Categories**:
  - 🟢 Green: Equal preferences
  - 🔴 Red: Extreme preferences (focus on one task)
  - 🔵 Blue: Balanced preferences

## Visualization Suite

### Before (6 visualization types):
1. Performance Heatmap
2. Radar Charts
3. Task Interference Matrix
4. Pareto Frontiers (6 plots for 4 tasks)
5. Preference Alignment
6. W&B Bar Charts

### After (9 visualization types + exports):
1. Performance Heatmap *(improved labels)*
2. Radar Charts *(improved labels)*
3. Task Interference Matrix *(improved labels)*
4. Pareto Frontiers *(cleaner, no hypervolume/nadir, better labels)*
5. Preference Alignment *(improved labels, clarified normalization)*
6. **Parallel Coordinates** *(NEW)*
7. **Distance to Utopia** *(NEW)*
8. **Performance Recovery Analysis** *(NEW - absolute & percentage)*
9. W&B Bar Charts
10. **Comprehensive Results Export** *(NEW - JSON & CSV)*

## Files Modified

1. `src/visualization/pareto.py` - Commented out hypervolume/nadir, improved labels
2. `src/visualization/plots.py` - Added formatting utilities, new visualizations
3. `src/visualization/generator.py` - Integrated new plots, added export function
4. `src/benchmarks/poc/run.py` - Integrated export into benchmark

## Usage

The visualizations and exports are generated automatically when running the benchmark:

```bash
python main.py
```

Results will be saved in:
```
outputs/proof_of_concept/<timestamp>/visualizations/
├── performance_heatmap.png
├── radar_charts/
├── task_interference.png
├── pareto_frontiers/
├── preference_alignment/
├── parallel_coordinates.png          # NEW
├── distance_to_utopia.png            # NEW
├── performance_recovery.png          # NEW
├── comprehensive_results.json        # NEW
├── comprehensive_results.csv         # NEW
├── summary_statistics.csv            # NEW
└── utopia_distances.csv             # NEW
```

## Research Foundations

Improvements based on best practices from:
- [Visualization of Pareto Front Points when Solving Multi-objective Optimization Problems](https://www.researchgate.net/publication/262698399_Visualization_of_Pareto_Front_Points_when_Solving_Multi-objective_Optimization_Problems)
- [Performance indicators in multiobjective optimization](https://hal.science/hal-03048871/document)
- [Multi-objective optimization with Pareto front – d3VIEW](https://www.d3view.com/multi-objective-optimization-with-pareto-front/)
- [Multiobjective optimization - OpenMDAO](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/multiobjective.html)

## Key Insights

### What the Visualizations Tell You:

1. **Pareto Frontiers**: Show trade-offs between pairs of tasks
   - Utopia point (gold diamond): Theoretical best on both tasks
   - Fine-tuned models (colored triangles): Actual single-task performance
   - Pareto front (red line): Best achievable trade-offs with current method

2. **Parallel Coordinates**: Quick comparison across ALL tasks
   - Spot preference vectors that generalize well
   - Identify task-specific vs. robust solutions

3. **Distance to Utopia**: Which preference weighting works best
   - Lower distance = closer to ideal performance
   - Helps choose optimal preference vector

4. **Performance Recovery**: Tests if multi-task model meets expectations
   - Expected = preference_weight × single_task_performance
   - Green bars (≥100%): Positive transfer - model exceeds expectations
   - Orange bars (80-100%): Minimal interference - close to expected
   - Red bars (<80%): Negative transfer - underperforming
   - Reveals which tasks benefit/suffer from multi-task learning

5. **Performance Heatmap**: Overview of all results
   - Identify weak tasks (red columns)
   - See how preferences affect each task

6. **Task Interference**: Which tasks conflict/synergize
   - Positive correlation (red): Tasks help each other
   - Negative correlation (blue): Tasks interfere

## Next Steps / Suggestions

### Potential Future Enhancements:

1. **Interactive Visualizations**: Use Plotly/Dash for interactive exploration
2. **3D Pareto Surfaces**: For visualizing 3-task trade-offs
3. **Animation**: Show how solutions evolve across training epochs
4. **Clustering Analysis**: Group similar preference vectors
5. **Sensitivity Analysis**: How sensitive are results to preference weights?

## Recent Updates (From todos.txt)

### Preference Alignment Clarification ✅
**Issue**: The normalization method for "achieved" weights wasn't clear, and "weight" wasn't the right label.

**Solution**:
- Changed Y-axis label from "Weight" to "Relative Weight"
- Updated legend to show "Achieved (÷ sum)" to clarify normalization
- Added explanatory text box: "Achieved normalized by sum of all task scores"
- Better axis labels: "Requested Preference Weight" vs "Achieved Score (÷ sum)"

### Smart Pareto Point Labeling ✅
**Issue**: Labeling all points on Pareto frontier would clutter the visualization.

**Solution**: Implemented intelligent selective labeling:
1. **Always label**:
   - Closest to utopia point (yellow box, most important)
   - Best performance on X-axis task (light blue box)
   - Best performance on Y-axis task (light coral box)

2. **Display count**: Shows "# other Pareto-optimal solution(s)" for unlabeled points

3. **Benefits**:
   - No visual clutter
   - Highlights most important trade-off solutions
   - User knows how many points exist even if not labeled

### Performance Recovery Analysis ✅
**Motivation**: Need to test the hypothesis that multi-task models should achieve at least a proportional fraction of single-task performance based on preference weights.

**Hypothesis**: Given a preference vector with weight `w_i` for task `i`, the multi-task model should achieve at least `w_i × single_task_performance_i` on that task.

**Implementation**:
1. **Formula**:
   - Expected performance: `preference_weight × single_task_performance`
   - Recovery rate: `(achieved / expected) × 100`

2. **Two-panel visualization**:
   - **Left**: Absolute values (expected vs achieved bars)
   - **Right**: Recovery percentage with color coding

3. **Color Coding**:
   - 🟢 Green (≥100%): Positive transfer - exceeds expectation
   - 🟠 Orange (80-100%): Minimal interference - close to expected
   - 🔴 Red (<80%): Negative transfer - underperforming

4. **Key Insights**:
   - Identifies which tasks benefit from multi-task learning (green)
   - Reveals negative task interference (red)
   - Shows if preference weights effectively guide optimization
   - Helps validate multi-objective optimization approach

**Scientific Validity**: This is a valid baseline because:
- Tests a reasonable expectation: proportional performance recovery
- Does NOT assume linear scaling of actual performance
- Does NOT ignore task correlations (interference/synergy shown by deviations)
- Provides interpretable metric for comparing multi-task vs single-task models

**File**: `src/visualization/plots.py` (lines 770-937), integrated in `src/visualization/generator.py` (lines 156-167)

## Notes

- All visualizations use 300 DPI for publication quality
- Both PNG and PDF formats can be generated (currently PNG only)
- Color schemes are colorblind-friendly
- All labels use consistent formatting
- Smart labeling prevents clutter while maintaining information
