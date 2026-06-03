# Generalized Hough Transform

## Introduction
The **Generalized Hough Transform (GHT)** is a powerful technique in computer vision used for object recognition. While the standard Hough Transform is typically used to detect regular, analytically defined shapes like lines, circles, and ellipses, the *Generalized* Hough Transform allows for the detection of arbitrary, complex, and non-analytical shapes.

This is incredibly useful in scenarios where the object being searched for lacks a simple mathematical equation. By relying on a template image of the object, the GHT is robust against partial occlusion, noise, and minor deformations, making it a highly reliable tool for pattern matching and object detection in real-world image processing tasks.

## Theory
The Generalized Hough Transform operates in two main phases: building a Reference Table (R-Table) from a template, and voting in an Accumulator space for the target image.

### 1. Building the R-Table
Instead of using equations, the GHT represents a shape using an **R-Table**. The algorithm takes a template image of the desired object:
1. **Edge Detection:** The shape's boundary is extracted using an edge detector (like Canny).
2. **Gradient Calculation:** The gradient magnitude and orientation ($\theta$) are computed for every edge pixel.
3. **Reference Point:** A reference point $(x_c, y_c)$ is chosen, typically the centroid of the template.
4. **Populating the Table:** For every edge pixel, a spatial vector connecting it to the reference point is calculated (represented by distance $r$ and angle $\alpha$). This vector is stored in the R-Table, indexed by the pixel's gradient orientation $\theta$.

### 2. Accumulator Voting Process
Once the R-Table is constructed, the algorithm searches for the shape in the main target image:
1. **Edge Detection and Gradients:** Edges and gradient orientations are computed for the target image.
2. **Voting:** For every edge pixel in the target image with a gradient orientation $\theta$, the algorithm looks up the corresponding $\theta$ entry in the R-Table.
3. **Candidate Centers:** For each vector $(r, \alpha)$ found in the table entry, a candidate reference center is calculated in the target image, and a vote is cast in an Accumulator array at that coordinate.
4. **Transformations:** If the object might be scaled or rotated, the vectors $(r, \alpha)$ are adjusted mathematically according to the tested scales and rotations, and votes are cast in a higher-dimensional Accumulator space.
5. **Detection:** The locations (and corresponding scales/rotations) with the highest number of votes in the Accumulator indicate the most likely positions of the detected object.

## Practical Approach
This repository provides a highly optimized Python implementation of the Generalized Hough Transform. Key practical improvements include:

* **Continuous Gradients:** Rather than relying solely on binary edge maps, continuous gradient orientations are calculated directly from grayscale images using Sobel filters, leading to much more accurate angle indexing.
* **Vectorized Accumulator with NumPy Broadcasting:** Instead of slow iterative voting loops, this implementation heavily leverages NumPy broadcasting to compute candidate centers and cast votes simultaneously across multiple pixels, scales, and rotations.
* **Optimized OpenCV Convolutions:** Manual convolution loops for gradient calculation have been replaced with `cv2.filter2D`, drastically reducing execution time while maintaining the required float precision and zero-padding logic.

## Dependencies
To run this project, the following Python libraries are required:
* `numpy`
* `opencv-python` (cv2)
* `matplotlib`

## Usage

The script is executed via the command line.

**Note on Headless Environments:** The script detects non-interactive matplotlib backends (e.g. `Agg`) automatically and simply skips the on-screen display while still saving the output figure. If you want to force a headless backend, you can still set it explicitly:
```bash
export MPLBACKEND=Agg
```

**Output Directory:** The script automatically creates the `output/` directory if it does not exist, so no manual setup is required.

### Syntax
```bash
python3 generalizedHoughTransform.py <mainImageName> <referenceImageName> [OPTIONS]
```

### Positional Arguments
* `mainImageName` : Path to the main image where the object is to be searched.
* `referenceImageName` : Path to the template/reference image of the object.

### Optional Arguments
* `--threshold_ratio` : Float (default: `0.8`). The ratio relative to the maximum vote to consider as a valid match. Lower values yield more matches.
* `--scales` : List of floats (default: `[1.0]`). Scale factors to test. Example: `--scales 0.8 1.0 1.2`
* `--rotations` : List of floats (default: `[0.0]`). Rotation angles in degrees to test. Example: `--rotations 0 90 180 270`
* `--canny_low` : Integer (default: `100`). Low threshold for the Canny edge detector.
* `--canny_high` : Integer (default: `200`). High threshold for the Canny edge detector.
* `--theta_bin` : Integer (default: `1`). Angular bin size in degrees used to match gradient orientations between the template and the target. A value of `1` keeps the original per-degree behavior; larger values (e.g. `5`) add angular tolerance and improve robustness to noise and to shapes that are not pixel-identical to the template.
* `--rotations_range` : Three floats `START STOP STEP` (no default). Convenience syntax to generate the list of rotations as `np.arange(START, STOP, STEP)`. `STOP` is exclusive, so `--rotations_range 0 360 5` produces `0, 5, 10, ..., 355`. If provided, this option takes precedence over `--rotations`.
* `--scales_range` : Three floats `START STOP STEP` (no default). Convenience syntax to generate the list of scales as `np.arange(START, STOP, STEP)`. `STOP` is exclusive, so `--scales_range 0.8 1.3 0.1` produces `0.8, 0.9, 1.0, 1.1, 1.2`. If provided, this option takes precedence over `--scales`.

## Example

```bash
# Run the script searching for 'template.png' inside 'image.png',
# checking scales 0.8, 1.0, 1.2 and rotations 0, 90, 180, 270 degrees.
python3 generalizedHoughTransform.py image.png template.png \
  --scales 0.8 1.0 1.2 \
  --rotations 0 90 180 270 \
  --canny_low 50 \
  --canny_high 150
```

Or, equivalently, using the range syntax — convenient when many values are needed (e.g. a full 360° rotation sweep at 5° steps):

```bash
python3 generalizedHoughTransform.py image.png template.png \
  --scales_range 0.8 1.3 0.1 \
  --rotations_range 0 360 5
```

*Expected Behavior:* The script will process the images, build the R-Table, and calculate votes across the specified 4D space (X, Y, Scale, Rotation). It will output the number of matches found to the console (and, when more than one scale or rotation is tested, a one-line summary of the search-space size) and save a visual plot of the original image, template, accumulator heat map, and the matched locations to `output/output.png`.
