import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def buildingReferenceTable(template_gray, template_edges):
    """
    Build the R-Table (Reference Table) for the Generalized Hough Transform.

    This function calculates continuous gradients on the grayscale template to obtain
    accurate edge orientations, while restricting the reference point calculations
    exclusively to the spatial coordinates defined by the binary edge map. For each 
    edge pixel, it computes the spatial vector (radial distance 'r' and angle 'alpha') 
    relative to the template's centroid and indexes it by its quantized gradient orientation.

    Parameters:
    template_gray (numpy.ndarray): The grayscale template image used for continuous gradient calculation.
    template_edges (numpy.ndarray): The binary edge map (e.g., Canny output) used as a spatial mask.

    Returns:
    dict: The Reference Table mapping quantized gradient orientations (0-359 degrees)
          to a list of (r, alpha) tuples pointing to the template's centroid.
    """
    
    # Convertiamo in scala di grigi se non lo è già
    if len(template_gray.shape) == 3:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)

    Ix, Iy = sobel_filter(template_gray)
    G, theta = gradient_intensity(Ix, Iy)

    # Define the center point of the template
    xc, yc = template_gray.shape[0] // 2, template_gray.shape[1] // 2

    # Initialize the reference table
    reference_table = {}
    edge_pixels = np.argwhere(template_edges > 0)
    for x, y in edge_pixels:
        r = np.sqrt((xc - x)**2 + (yc - y)**2)
        alpha = np.arctan2(yc - y, xc - x)

        orientation = int(np.round(np.degrees(theta[x, y]))) % 360
        if orientation not in reference_table:
            reference_table[orientation] = []
        reference_table[orientation].append((r, alpha))
    return reference_table


def calculate_accumulator(image_gray, image_edges, reference_table, scales=[1.0], rotations=[0.0]):
    """
    Calculate the accumulator array for the given image and reference table.

    Parameters:
    image_gray (numpy.ndarray): The input grayscale image.
    image_edges (numpy.ndarray): The Canny edges map.
    reference_table (dict): The reference table.
    scales (list): List of scale factors.
    rotations (list): List of rotation angles in degrees.

    Returns:
    numpy.ndarray: The accumulator array.
    """
    Ix, Iy = sobel_filter(image_gray)
    G, theta = gradient_intensity(Ix, Iy)

    is_2d = scales == [1.0] and rotations == [0.0]

    # Initialize the accumulator array
    if is_2d:
        accumulator = np.zeros_like(image_gray, dtype=np.float64)
    else:
        accumulator = np.zeros((image_gray.shape[0], image_gray.shape[1], len(scales), len(rotations)), dtype=np.float64)

    # Get the indices of the edge pixels
    edge_pixels = np.argwhere(image_edges > 0)

    # Get the orientation for all edge pixels, quantized to integer degrees
    orientations = np.round(np.degrees(theta[edge_pixels[:, 0], edge_pixels[:, 1]])).astype(int) % 360
    unique_orientations = np.unique(orientations)

    scales_ary = np.array(scales)

    for o in unique_orientations:
        # Mask for edge pixels with orientation `o`
        mask = orientations == o
        pixels = edge_pixels[mask]

        r_list = []
        alpha_new_list = []
        r_idx_list = []

        for r_idx, phi_deg in enumerate(rotations):
            # Calculate the expected template orientation
            o_tpl = int((o - phi_deg) % 360)
            if o_tpl in reference_table:
                r_alpha = np.array(reference_table[o_tpl])
                r = r_alpha[:, 0]
                alpha = r_alpha[:, 1]

                # Apply rotation to alpha
                phi_rad = np.radians(phi_deg)
                alpha_new = alpha + phi_rad

                r_list.append(r)
                alpha_new_list.append(alpha_new)
                r_idx_list.append(np.full(len(r), r_idx, dtype=int))

        if not r_list:
            continue

        # Concatenate all valid vectors across all rotations
        r_cat = np.concatenate(r_list)
        alpha_new_cat = np.concatenate(alpha_new_list)
        r_idx_cat = np.concatenate(r_idx_list)

        # Apply scales (broadcasting over S scales and K vectors)
        # r_new: shape (S, K)
        r_new = r_cat[None, :] * scales_ary[:, None]
        # alpha_new_expanded: shape (S, K)
        alpha_new_expanded = np.broadcast_to(alpha_new_cat[None, :], r_new.shape)

        # Calculate dx and dy for all reference points across all scales and rotations
        dx = (r_new * np.cos(alpha_new_expanded)).astype(int)
        dy = (r_new * np.sin(alpha_new_expanded)).astype(int)

        # Broadcast to calculate candidate center points for all pixels matching orientation `o`
        # pixels: (N, 2), dx/dy: (S, K) -> xc/yc: (N, S, K)
        xc = pixels[:, 0, None, None] + dx[None, :, :]
        yc = pixels[:, 1, None, None] + dy[None, :, :]

        xc = xc.ravel()
        yc = yc.ravel()

        if is_2d:
            valid_mask = (xc >= 0) & (xc < image_gray.shape[0]) & (yc >= 0) & (yc < image_gray.shape[1])
            np.add.at(accumulator, (xc[valid_mask], yc[valid_mask]), 1)
        else:
            # Create s_idx array: shape (S, K)
            s_idx_cat = np.arange(len(scales))[:, None]
            s_idx_cat = np.broadcast_to(s_idx_cat, dx.shape)

            # Expand r_idx_cat: shape (S, K)
            r_idx_expanded = np.broadcast_to(r_idx_cat[None, :], dx.shape)

            # Broadcast indices to match (N, S, K)
            s_idx_final = np.broadcast_to(s_idx_cat[None, :, :], (len(pixels), len(scales), len(r_cat))).ravel()
            r_idx_final = np.broadcast_to(r_idx_expanded[None, :, :], (len(pixels), len(scales), len(r_cat))).ravel()

            valid_mask = (xc >= 0) & (xc < image_gray.shape[0]) & (yc >= 0) & (yc < image_gray.shape[1])
            np.add.at(accumulator, (xc[valid_mask], yc[valid_mask], s_idx_final[valid_mask], r_idx_final[valid_mask]), 1)

    return accumulator


def find_best_match(accumulator, scales=[1.0], rotations=[0.0]):
    """
    Find the location with the highest vote in the accumulator.

    Parameters:
    accumulator (numpy.ndarray): The accumulator array.
    scales (list): List of scale factors.
    rotations (list): List of rotation angles in degrees.

    Returns:
    tuple: The location with the highest vote (x, y, scale, rotation).
    """
    best_matched_center = np.unravel_index(accumulator.argmax(), accumulator.shape)
    if accumulator.ndim == 2:
        return (best_matched_center[0], best_matched_center[1], scales[0], rotations[0])
    else:
        return (best_matched_center[0], best_matched_center[1], scales[best_matched_center[2]], rotations[best_matched_center[3]])


def find_all_matches(accumulator, threshold_ratio=0.8, scales=[1.0], rotations=[0.0]):
    """
    Find all locations with votes in the accumulator.

    Parameters:
    accumulator (numpy.ndarray): The accumulator array.
    threshold_ratio (float): The threshold ratio for finding all matches.
    scales (list): List of scale factors.
    rotations (list): List of rotation angles in degrees.

    Returns:
    list: The locations with votes, each as (x, y, scale, rotation).
    """
    threshold = threshold_ratio * np.max(accumulator)
    matched_centers = np.argwhere(accumulator >= threshold)

    results = []
    if accumulator.ndim == 2:
        for c in matched_centers:
            results.append((c[0], c[1], scales[0], rotations[0]))
    else:
        for c in matched_centers:
            results.append((c[0], c[1], scales[c[2]], rotations[c[3]]))

    return results


def convolution_padding(image, kernel):
    """
    Convolve an image with a kernel using OpenCV.

    Parameters:
    image (numpy.ndarray): The input image.
    kernel (numpy.ndarray): The kernel.

    Returns:
    numpy.ndarray: The convolved image.
    """
    # PERFORMANCE OPTIMIZATION:
    # Replacing manual nested loops with cv2.filter2D.
    #
    # Why:
    # 1. Manual loops in Python are extremely slow for pixel-wise operations because
    #    each iteration incurs high interpreter overhead.
    # 2. cv2.filter2D is implemented in C++ and highly optimized using SIMD (Single Instruction, Multiple Data)
    #    and multi-threading where possible.
    # 3. This change can lead to a speedup of 100x-1000x for typical image sizes.

    # Original manual implementation (commented for reference):
    # kernel_height, kernel_width = kernel.shape
    # pad_height, pad_width = kernel_height // 2, kernel_width // 2
    # padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # output = np.zeros_like(image, dtype=np.float64)
    # for x in range(image.shape[0]):
    #     for y in range(image.shape[1]):
    #         region = padded_image[x:x + kernel_height, y:y + kernel_width]
    #         output[x, y] = np.sum(region * kernel)

    # The optimized implementation below preserves the exact same behavior:
    # - borderType=cv2.BORDER_CONSTANT with value 0 handles the padding like np.pad.
    # - cv2.filter2D performs cross-correlation, matching the manual logic of multiplying the kernel with the region without flipping it.
    output = cv2.filter2D(image.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return output


def sobel_filter(image):
    """
    Apply Sobel filter to the image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: The gradients in the x and y directions.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    Ix = convolution_padding(image, Kx)
    Iy = convolution_padding(image, Ky)

    return Ix, Iy


def gradient_intensity(Ix, Iy):
    """
    Calculate the gradient intensity of the image.

    Parameters:
    Ix (numpy.ndarray): The gradient in the x direction.
    Iy (numpy.ndarray): The gradient in the y direction.

    Returns:
    tuple: The gradient intensity and the orientation.
    """
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)


def plotBestMatches(image, template, accumulator, matched_centers, best_match):
    """
    Plot the original image, template, accumulator, and matched locations.

    Parameters:
    image (numpy.ndarray): The original image.
    template (numpy.ndarray): The template.
    accumulator (numpy.ndarray): The accumulator array.
    matched_centers (list): The matched centers.
    best_match (tuple): The best matched center.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original Image')

    ax[0, 1].imshow(template, cmap='gray')
    ax[0, 1].set_title('Template')

    if accumulator.ndim >= 3:
        # Display the maximum vote across all extra dimensions for each pixel
        ax[1, 0].imshow(np.max(accumulator, axis=tuple(range(2, accumulator.ndim))), cmap='hot')
    else:
        ax[1, 0].imshow(accumulator, cmap='hot')
    ax[1, 0].set_title('Accumulator')

    ax[1, 1].imshow(image, cmap='gray')
    for center in matched_centers:
        ax[1, 1].scatter(center[1], center[0], s=100, c='green', marker='x')
    ax[1, 1].scatter(best_match[1], best_match[0], s=100, c='red', marker='x')

    title = f'Matched Locations\nBest: scale={best_match[2]}, rot={best_match[3]}°' if len(best_match) >= 4 else 'Matched Locations'
    ax[1, 1].set_title(title)

    # Save the plot
    plt.savefig('output/output.png')

    plt.show()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generalized Hough Transform")

    # Add the arguments
    parser.add_argument('mainImageName', type=str, help='The main image file name')
    parser.add_argument('referenceImageName', type=str, help='The reference image file name')
    parser.add_argument('--threshold_ratio', type=float, default=0.8, help='The threshold ratio for finding all matches')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0], help='List of scales to use (default: 1.0)')
    parser.add_argument('--rotations', type=float, nargs='+', default=[0.0], help='List of rotations in degrees to use (default: 0.0)')
    
    # NUOVI PARAMETRI: Soglie per Canny Edge Detector
    parser.add_argument('--canny_low', type=int, default=100, help='Low threshold for Canny edge detector (default: 100)')
    parser.add_argument('--canny_high', type=int, default=200, help='High threshold for Canny edge detector (default: 200)')

    # Parse the arguments
    args = parser.parse_args()

    mainImageName = args.mainImageName
    referenceImageName = args.referenceImageName

    referenceImage = cv2.imread(referenceImageName)
    mainImage = cv2.imread(mainImageName)

    # Converting the RGB image to a grayscale image.
    referenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_RGB2GRAY)
    mainImage = cv2.cvtColor(mainImage, cv2.COLOR_RGB2GRAY)

    # Perform edge detection on the images using CLI parameters
    template = cv2.Canny(referenceImage, args.canny_low, args.canny_high)
    image = cv2.Canny(mainImage, args.canny_low, args.canny_high)

    reference_table = buildingReferenceTable(referenceImage, template)

    # Calculate the accumulator array
    accumulator = calculate_accumulator(mainImage, image, reference_table, args.scales, args.rotations)

    # Find the best matched location
    best_matched_center = find_best_match(accumulator, args.scales, args.rotations)

    # Find all matched locations
    matched_centers = find_all_matches(accumulator, args.threshold_ratio, args.scales, args.rotations)
    print(f"Trovati {len(matched_centers)} match!")

    # Plot the results
    plotBestMatches(mainImage, referenceImage, accumulator, matched_centers, best_matched_center)
