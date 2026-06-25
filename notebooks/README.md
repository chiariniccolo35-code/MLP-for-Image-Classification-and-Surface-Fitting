# Notebooks

Two independent Jupyter notebooks (originally developed on Google Colab), corresponding to the two chapters of the report.

## MLP_Image_Classification.ipynb

**164 cells.** Implements the full Chapter 1 hyperparameter study for Fashion-MNIST classification using a Multilayer Perceptron (Keras/TensorFlow).

**Contents:**
1. Library imports and Fashion-MNIST dataset loading (`tensorflow.keras.datasets`)
2. Data visualization (sample images per class)
3. MLP model definitions, varying:
   - Number of dense layers (5 / 10 / 15)
   - Neurons per layer (32 / 64 / 128)
   - Activation function (ReLU / Sigmoid / Tanh)
   - Batch size (32 / 128 / 256 / 512)
   - Number of epochs (50 / 100)
   - Optimizer (SGD / ADAM)
4. Training loops and evaluation (training/validation/test loss & accuracy)
5. Plotting of loss/accuracy curves for every configuration tested

**How to run:**

```bash
jupyter notebook MLP_Image_Classification.ipynb
```

No external data files are required — Fashion-MNIST is downloaded automatically via Keras on first run.

**Note:** the notebook is structured as a sequence of largely independent experiment blocks (one per hyperparameter study, matching the report's Sections 1.1–1.7) rather than a single parametrized training function — re-run the relevant cells to reproduce a specific configuration's results.

## SIREN_3D_Surface_Fitting.ipynb

**44 cells.** Implements Chapter 2: a SIREN (Sinusoidal Representation Network) trained to fit a Signed Distance Function (SDF) and reconstruct the Stanford bunny from its point cloud.

**Contents:**
1. Package installation (`pyvista`, `xvfb`, `libgl1-mesa-glx` for headless 3D visualization) — Colab-specific setup cell
2. Imports: `pyvista`, `tensorflow`, `plyfile` (point-cloud/mesh I/O), `skimage.measure` (marching cubes for surface extraction)
3. Point-cloud data loading (bunny)
4. SIREN model definition (sine activations, `ω₀` frequency scaling, custom weight initialization as derived in report Section 2.2)
5. Training loop fitting the SDF from the point cloud
6. Surface extraction (marching cubes) and 3D visualization via `pyvista`
7. Comparison across training lengths (1,000 / 5,000 / 10,000 epochs) and across the two initialization schemes discussed in the report

**How to run:**

```bash
jupyter notebook SIREN_3D_Surface_Fitting.ipynb
```

**Requirements (beyond standard TensorFlow):**
```bash
pip install pyvista plyfile scikit-image
```

If running outside Colab/headless environments, the `xvfb`/`libgl1-mesa-glx` virtual-display setup cells can be skipped on a machine with a normal display; `pv.start_xvfb()` should only be called when no display is available.

## Notebook ↔ Report Section Mapping

| Notebook | Report Section(s) |
|---|---|
| `MLP_Image_Classification.ipynb` | Chapter 1 (1.1 – 1.7) |
| `SIREN_3D_Surface_Fitting.ipynb` | Chapter 2 (2.1 – 2.3) |
