# Ultimate Conics Generator

An interactive web-based tool for exploring generalized conics through various distance metrics and dimensional modes. This application allows users to place multiple focus points (foci) and visualize the resulting curves or surfaces where the weighted sum of distances to those foci is constant.

## Features

### Dimensional Modes
* **2D Mode:** Visualizes classic curves such as circles, ellipses, hyperbolas, and complex N-foci shapes.
* **3D Constant:** Generates isosurfaces where the weighted sum of distances equals a constant $K$.
* **3D Variable:** Generates surfaces where the weighted sum of distances equals the $Z$ coordinate, allowing for the creation of cones and saddle surfaces.

### Distance Metrics
* **Euclidean:** Standard $L_2$ norm representing straight-line distance.
* **Manhattan:** $L_1$ norm representing taxicab geometry.
* **Chebyshev:** $L_\infty$ norm representing chessboard distance.
* **Minkowski:** A generalized metric that interpolates between Manhattan ($P=1$), Euclidean ($P=2$), and Chebyshev ($P \to \infty$) based on a user-defined power.
* **Angular:** Radial patterns modified by azimuthal angles to create star and rose shapes.



### Functionality
* **Auto-Constant Calculation:** Includes an algorithm to determine an optimal $K$ value based on the spatial distribution of the foci.
* **Multi-Branch Support:** Correctly renders both branches of hyperbolas by plotting both positive and negative levels of the distance field.
* **Vectorized Processing:** Utilizes NumPy broadcasting to ensure high-performance rendering and responsiveness.



## Mathematics

The application defines a scalar field $f(P)$ based on the position of $n$ foci ($F$):

$$f(P) = \sum_{i=1}^{n} w_i \cdot d(P, F_i)$$

Where:
* $P$ is a point in the coordinate grid.
* $w_i$ is the weight (sign) assigned to the focus ($+1$ or $-1$).
* $d$ is the selected distance metric.

In **3D Variable** mode, the surface is defined by the set of points where:
$$f(P) = Z$$

   cd ultimate-conics
