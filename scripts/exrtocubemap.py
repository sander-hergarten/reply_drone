#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converts an Equirectangular EXR/Image file to a vertically stacked Cubemap.
Output format can be EXR (HDR) or PNG (LDR with optional gamma correction).
Automatically determines face size if not provided (--size width/4).
Includes fallback logic for reading input files with different imageio plugins.
Handles cases where imageio reads input as a 4D array (1, H, W, C).
Uses 'reflect' mode for map_coordinates boundary handling.
"""

import numpy as np
import imageio.v3 as iio
import imageio
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import sys
from pathlib import Path  # Using pathlib for cleaner path handling
import tyro  # Import tyro
from typing import Literal  # For type hinting choices


def equirectangular_to_cubemap_face(
    equi_img: np.ndarray, face: str, face_size: int
) -> np.ndarray:
    """
    Generates one face of a cubemap from an equirectangular image.

    Args:
        equi_img: The equirectangular input image (H, W, C), expected to be float.
        face: The face to generate ('+x', '-x', '+y', '-y', '+z', '-z').
        face_size: The edge length of the cubemap face in pixels.

    Returns:
        The generated cubemap face (face_size, face_size, C) as float.
    """
    if equi_img.ndim != 3:
        raise ValueError(
            f"equirectangular_to_cubemap_face expects a 3D array (H, W, C), but got shape {equi_img.shape}"
        )
    if not np.issubdtype(equi_img.dtype, np.floating):
        # Ensure input is float for calculations, conversion should happen before calling this
        raise ValueError(
            f"equirectangular_to_cubemap_face expects float input, got {equi_img.dtype}"
        )

    equi_h, equi_w, channels = equi_img.shape
    dtype = equi_img.dtype  # Should be float

    # Create grid coordinates for the face
    uv = np.linspace(-1.0, 1.0, face_size)
    u, v = np.meshgrid(uv, uv)  # v ranges top-to-bottom, u ranges left-to-right

    # Map face coordinates (u, v) to 3D direction vectors (x, y, z)
    # Standard OpenGL/Blender cubemap layout
    if face == "+x":
        x = np.ones_like(u)
        y = -v
        z = -u
    elif face == "-x":
        x = -np.ones_like(u)
        y = -v
        z = u
    elif face == "+y":
        x = u
        y = np.ones_like(u)
        z = v
    elif face == "-y":
        x = u
        y = -np.ones_like(u)
        z = -v
    elif face == "+z":  # Back face
        x = u
        y = -v
        z = np.ones_like(u)
    elif face == "-z":  # Front face
        x = -u
        y = -v
        z = -np.ones_like(u)
    else:
        raise ValueError("Invalid face identifier")

    # Normalize the direction vectors
    norm = np.sqrt(x * x + y * y + z * z)
    # Avoid division by zero for potential corner cases, though unlikely with linspace
    norm[norm == 0] = 1e-6
    x /= norm
    y /= norm
    z /= norm

    # Convert 3D direction vector (x, y, z) to spherical coordinates (phi, theta)
    phi = np.arctan2(x, z)  # Azimuth [-pi, pi]
    theta = np.arcsin(y)  # Inclination [-pi/2, pi/2]

    # Convert spherical coordinates (phi, theta) to equirectangular image coordinates (px, py)
    px = (phi / (2 * np.pi) + 0.5) * equi_w
    py = (-theta / np.pi + 0.5) * equi_h

    # --- Interpolation ---
    output_face = np.zeros((face_size, face_size, channels), dtype=dtype)
    coords = np.stack((py, px), axis=0)  # Coordinate array for map_coordinates

    # Interpolate each channel
    for c in range(channels):
        output_face[..., c] = map_coordinates(
            equi_img[..., c],
            coords,
            order=1,  # Bilinear interpolation
            mode="reflect",  # Use single mode 'reflect'
            prefilter=False,
        )

    return output_face


def convert_equi_to_cubemap(
    input_path: Path,
    output_path: Path,
    size: int | None = None,
    read_plugin: str | None = None,
    output_type: Literal["exr", "png"] = "exr",  # Choose output type
    gamma: float = 2.2,  # Gamma correction value for PNG output
):
    """
    Main conversion logic: Loads an equirectangular image, converts it to a
    vertically stacked cubemap, and saves the result.

    Args:
        input_path: Path to the input Equirectangular image file (e.g., .exr, .hdr, .png).
        output_path: Path for the output vertical cubemap file. Extension will be adjusted based on output_type.
        size: Edge size of each cubemap face in pixels. If None, defaults to input_width / 4.
        read_plugin: Specific imageio plugin to use for reading. Tries auto-detect if None.
        output_type: Format for the output file ('exr' or 'png'). Default is 'exr'.
        gamma: Gamma correction value to apply when output_type is 'png'. Default is 2.2.
    """

    imageio.plugins.freeimage.download()
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # --- Adjust output path extension ---
    output_suffix = f".{output_type}"
    if output_path.suffix.lower() != output_suffix:
        print(
            f"Adjusting output path extension to {output_suffix}: {output_path.with_suffix(output_suffix)}"
        )
        output_path = output_path.with_suffix(output_suffix)
    # ---

    print(f"Loading input image: {input_path}")
    equi_image_float = None  # We'll store the float version here
    used_plugin = read_plugin

    try:
        # Attempt to read using the specified plugin or auto-detect
        print(
            f"Attempting to read with plugin: {used_plugin if used_plugin else 'auto-detect'}..."
        )
        if used_plugin is not None:
            raw_image = iio.imread(input_path, plugin=used_plugin)
        else:
            raw_image = iio.imread(input_path)

        print(
            f"Raw image shape read by imageio: {raw_image.shape}, dtype: {raw_image.dtype}"
        )
        # --- Check dimensions and squeeze if necessary ---
        if raw_image.ndim == 4 and raw_image.shape[0] == 1:
            print(f"Detected 4D array with leading dimension 1. Squeezing to 3D.")
            processed_image = np.squeeze(raw_image, axis=0)
        elif raw_image.ndim == 3 or raw_image.ndim == 2:
            processed_image = raw_image  # Shape is already okay
        else:
            raise ValueError(
                f"Read image, but dimensions are unexpected: {raw_image.shape}"
            )

        # --- Convert to float32 and normalize if necessary ---
        if not np.issubdtype(processed_image.dtype, np.floating):
            print(
                f"Input image dtype is {processed_image.dtype}. Converting to float32."
            )
            if np.issubdtype(processed_image.dtype, np.integer):
                max_val = np.iinfo(processed_image.dtype).max
                # Avoid division by zero if max_val is 0 (though unlikely for images)
                if max_val > 0:
                    equi_image_float = processed_image.astype(np.float32) / max_val
                    print(f"Normalized integer input assuming max value {max_val}.")
                else:
                    equi_image_float = processed_image.astype(np.float32)
                    print(
                        "Integer input with max value 0, converted to float without normalization."
                    )
            else:
                equi_image_float = processed_image.astype(np.float32)
        else:
            equi_image_float = processed_image.astype(np.float32)  # Ensure it's float32

        # --- Handle grayscale ---
        if equi_image_float.ndim == 2:
            print("Input is grayscale. Adding channel dimension.")
            equi_image_float = np.expand_dims(equi_image_float, axis=-1)

        if equi_image_float.ndim != 3:  # Final check
            raise ValueError(
                f"Processed image dimensions are unexpected: {equi_image_float.shape}"
            )

        if used_plugin is None:
            try:
                metadata = iio.immeta(input_path)
                used_plugin = metadata.get("plugin", "auto-detected")
            except Exception:
                used_plugin = "auto-detected (meta failed)"
        print(f"Successfully read and processed using plugin: {used_plugin}")

    except Exception as e_initial:
        print(f"Failed to read with plugin '{used_plugin}': {e_initial}")
        # Fallback logic (simplified, assumes float conversion happens after successful read)
        alternative_plugins = [
            "EXR-FI",
            "EXR-PIL",
            "HDR-FI",
            "PNG-FI",
            "JPEG-PIL",
            "TIFF-FI",
        ]  # Add more common plugins
        if used_plugin is not None and used_plugin in alternative_plugins:
            alternative_plugins.remove(used_plugin)

        read_success = False
        for plugin in alternative_plugins:
            try:
                print(f"Attempting fallback read with plugin: {plugin}...")
                raw_image = iio.imread(input_path, plugin=plugin)
                print(
                    f"Raw image shape read by imageio (fallback): {raw_image.shape}, dtype: {raw_image.dtype}"
                )

                if raw_image.ndim == 4 and raw_image.shape[0] == 1:
                    processed_image = np.squeeze(raw_image, axis=0)
                elif raw_image.ndim == 3 or raw_image.ndim == 2:
                    processed_image = raw_image
                else:
                    raise ValueError(
                        f"Read image (fallback), but dimensions are unexpected: {raw_image.shape}"
                    )

                if not np.issubdtype(processed_image.dtype, np.floating):
                    if np.issubdtype(processed_image.dtype, np.integer):
                        max_val = np.iinfo(processed_image.dtype).max
                        if max_val > 0:
                            equi_image_float = (
                                processed_image.astype(np.float32) / max_val
                            )
                        else:
                            equi_image_float = processed_image.astype(np.float32)
                    else:
                        equi_image_float = processed_image.astype(np.float32)
                else:
                    equi_image_float = processed_image.astype(np.float32)

                if equi_image_float.ndim == 2:
                    equi_image_float = np.expand_dims(equi_image_float, axis=-1)

                if equi_image_float.ndim != 3:  # Final check
                    raise ValueError(
                        f"Processed image dimensions are unexpected (fallback): {equi_image_float.shape}"
                    )

                print(
                    f"Successfully read and processed using fallback plugin: {plugin}"
                )
                used_plugin = plugin
                read_success = True
                break  # Success
            except Exception as e_fallback:
                print(f"Fallback read with plugin '{plugin}' failed: {e_fallback}")

        if not read_success or equi_image_float is None:
            print(
                f"\nError: Could not read image file '{input_path}' with available imageio plugins."
            )
            print("Ensure the file is valid and imageio has suitable backends.")
            sys.exit(1)

    # --- Proceed with float image ---
    if equi_image_float is None:
        print("Error: equi_image_float is None after reading input.")
        sys.exit(1)
    equi_h, equi_w, _ = equi_image_float.shape
    print(
        f"Input image ready for processing: {equi_image_float.shape}, dtype={equi_image_float.dtype}"
    )

    # --- Determine face size ---
    if size is None:
        calculated_size = equi_w // 4
        print(
            f"--size not provided. Calculating size based on input width: {equi_w} / 4 = {calculated_size}"
        )
        size = calculated_size
    elif size <= 0:
        print(f"Error: Provided face size ({size}) must be positive.")
        sys.exit(1)
    else:
        print(f"Using provided face size: {size}")

    if size < 1:
        print(f"Warning: Calculated size is {size}, setting to minimum of 1.")
        size = 1

    print(f"Generating cubemap faces with size {size}x{size}...")
    face_order = ["+x", "-x", "+y", "-y", "+z", "-z"]
    cubemap_faces = []

    for face_name in tqdm(face_order, desc="Generating Faces"):
        # Pass the float image to the face generation function
        face_img = equirectangular_to_cubemap_face(equi_image_float, face_name, size)
        cubemap_faces.append(face_img)

    print("Stacking faces...")
    # Resulting stack is float32
    vertical_cubemap_f32 = np.vstack(cubemap_faces).astype(np.float32)
    print(
        f"Output cubemap dimensions: {vertical_cubemap_f32.shape}, dtype: {vertical_cubemap_f32.dtype}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Save the result based on output_type ---
    print(f"Saving output file: {output_path}")
    try:
        if output_type == "png":
            print(
                f"Output type is PNG. Applying gamma correction (gamma={gamma}), clamping, and converting to uint8."
            )
            # Clamp values to [0, 1] range first
            vertical_cubemap_clamped = np.clip(vertical_cubemap_f32, 0.0, 1.0)
            # Apply gamma correction
            # Add small epsilon to avoid potential log(0) issues if gamma is complex, although unlikely here
            epsilon = 1e-7
            vertical_cubemap_gamma_corrected = np.power(
                vertical_cubemap_clamped + epsilon, 1.0 / gamma
            )
            # Convert to uint8
            vertical_cubemap_uint8 = (
                np.clip(vertical_cubemap_gamma_corrected, 0.0, 1.0) * 255
            ).astype(np.uint8)
            # Save using the extension, let imageio choose plugin
            iio.imwrite(output_path, vertical_cubemap_uint8)
            print("PNG file saved.")
        elif output_type == "exr":
            print("Output type is EXR.")
            # Save float32 data using the extension
            iio.imwrite(output_path, vertical_cubemap_f32)
            print("EXR file saved.")
        else:
            # Should be caught by tyro Literal, but defensive check
            raise ValueError(f"Unsupported output_type: {output_type}")

    except ImportError as e:
        print(f"Error saving {output_type.upper()}. Imageio backend might be missing.")
        print(f"Import Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error writing {output_type.upper()} file '{output_path}': {e}")
        sys.exit(1)

    print("Done.")


def main():
    """
    Command-line entry point using tyro.
    Delegates to the core conversion function.
    """
    tyro.cli(convert_equi_to_cubemap)


if __name__ == "__main__":
    main()
