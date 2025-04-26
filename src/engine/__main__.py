import math
import os
import sys
import random
import string
from dataclasses import dataclass, field  # Make sure field is imported if needed
from typing import List, Tuple, Dict, Any, Optional  # Import typing helpers
from typing import cast

# Import numpy for seeding
import numpy as np

from engine.api import (
    AllBarcodeClassifiers,
    Engine,
    AllActions,
    FullGraphState,
    SingleBarcodeClassifier,
    SingleNodeState,
)

from panda3d.core import (
    Filename,
    PNMImage,
    NodePath,
    GeomVertexReader,
    Point3,
    Vec3,
)  # Added Point3, Vec3
from direct.showbase.ShowBase import ShowBase

import panda3d.core as p3d

import simplepbr

p3d.load_prc_file_data(
    "",
    "window-size 1024 768\n"
    "texture-minfilter mipmap\n"
    "texture-anisotropic-degree 16\n"
    "gl-finish true\n"
    "auto-flip true\n",
)


global INIT_SAMPLE_SIZE
global count
global all_actions
global full_graph
INIT_SAMPLE_SIZE = 10
count = 0
all_actions: AllActions = {}
full_graph: FullGraphState = {}


def take_screenshot(
    app_instance: ShowBase,
    filepath: Optional[str] = None,
    cam_pos: Optional[Tuple[float, float, float]] = None,
    cam_hpr: Optional[Tuple[float, float, float]] = None,
    return_data: bool = False,
) -> Optional[PNMImage | bool]:
    """Takes a screenshot using the provided ShowBase instance's camera and window."""
    original_pos = None
    original_hpr = None
    result = None
    win = app_instance.win
    cam = app_instance.cam
    graphics_engine = app_instance.graphicsEngine

    if not win or not cam or not graphics_engine:
        print(
            "Error: Panda3D ShowBase components (win, cam, graphicsEngine) not available."
        )
        return None

    # Store original camera state if we're moving it
    if cam_pos is not None or cam_hpr is not None:
        original_pos = cam.get_pos()
        original_hpr = cam.get_hpr()
        if cam_pos:
            cam.set_pos(*cam_pos)  # Unpack tuple
        if cam_hpr:
            cam.set_hpr(*cam_hpr)  # Unpack tuple

    # Ensure the frame with the potentially new camera view is rendered and ready
    # Multiple frames and sync might be needed depending on complexity/timing
    graphics_engine.renderFrame()
    graphics_engine.renderFrame()
    graphics_engine.syncFrame()  # Crucial for making sure pixels are ready
    win.triggerCopy()  # Request the copy operation

    # Retrieve screenshot data if requested
    if return_data:
        img = PNMImage()
        # Allow Panda some time to actually perform the copy-to-texture
        # Needed especially when getting data directly into PNMImage
        if graphics_engine.doYield():
            success = win.getScreenshot(img)
            if success and not img.is_null():  # Check if image data is valid
                result = img
                # print(f"Debug: Screenshot data captured: {img.getXSize()}x{img.getYSize()}")
            else:
                print(
                    f"Warning: Failed to capture screenshot data using getScreenshot(). Success={success}, Null={img.is_null()}"
                )
                result = None
        else:
            print(
                "Warning: graphics_engine.doYield() failed, cannot guarantee screenshot data."
            )
            result = None

    # Save to file if requested (regardless of return_data)
    if filepath:
        fname = Filename(filepath)
        fname.make_dir()  # Ensure directory exists
        # print(f"Debug: Attempting to save screenshot to {fname.to_os_specific()}...")
        save_success = win.saveScreenshot(fname)
        if not save_success:
            print(f"Warning: Failed to save screenshot to {filepath}")
        # If only saving (not returning data), result is the save success status
        if not return_data:
            result = save_success
        # print(f"Debug: saveScreenshot result: {save_success}")

    # Restore original camera state if it was changed
    if original_pos is not None:
        cam.set_pos(original_pos)
    if original_hpr is not None:
        cam.set_hpr(original_hpr)

    # Render a frame with the restored camera if needed, though usually not critical after screenshot
    # graphics_engine.renderFrame()

    return result


class App(ShowBase):
    # --- Class Members ---
    pipeline: Any  # Type hint for simplepbr pipeline if possible, else Any
    model_root: NodePath
    ambient_light_node: p3d.AmbientLight
    ambient_light_nodepath: NodePath
    anims: Optional[p3d.AnimControlCollection]  # Can be None if no anims

    def __init__(self):
        if len(sys.argv) < 2:
            print("Missing input file")
            print("Usage: python your_script_name.py path/to/model.glb")
            sys.exit(1)

        super().__init__()

        self.intialise_shortcuts()
        # --- Initialize Members ---
        self.anims = None  # Initialize anims

        self.pipeline = simplepbr.init()

        # --- Load Model ---
        self.load_model()

        # --- Camera Setup ---
        print("Calculating model bounds for camera setup...")
        min_pt, max_pt = Point3(), Point3()
        # Use calc_tight_bounds for potentially more accurate center/radius for visible geometry
        self.model_root.calc_tight_bounds(min_pt, max_pt)
        center = (min_pt + max_pt) / 2.0
        radius = 1.0  # Default radius

        if min_pt == max_pt:  # Check if bounds calculation failed or model is a point
            print(
                "Warning: Model bounds are zero or failed to compute. Using default camera position and radius."
            )
            center = p3d.Point3(0, 0, 0)
            # Fall back to getBounds which might be less tight but better than nothing
            bounds = self.model_root.getBounds()
            if not bounds.is_empty():
                center = bounds.get_center()
                radius = bounds.get_radius()
                if radius < 0.1:
                    radius = 0.5  # Clamp small radius
            else:
                print(
                    "Error: Both tight bounds and regular bounds failed. Using origin."
                )
        else:
            # Calculate radius based on the tight bounds
            radius = (max_pt - center).length()
            if radius < 0.1:
                print(
                    f"Warning: Model radius very small ({radius:.3f}). Clamping to 0.5 for camera distance."
                )
                radius = 0.5
            print(
                f"Model Tight Bounds Center: {center.getX():.2f}, {center.getY():.2f}, {center.getZ():.2f}"
            )
            print(f"Model Tight Bounds Radius: {radius:.2f}")

        self.enableMouse()  # Enable default mouse camera control

        # Sensible default camera position looking at the center
        default_distance = radius * 2.5  # Adjust multiplier as needed
        self.cam.set_pos(
            center + p3d.Vec3(0, -default_distance, radius * 0.5)
        )  # Position back and slightly up
        self.cam.look_at(center)  # Make it look at the center
        print(f"Default Camera Pos: {self.cam.get_pos()}, HPR: {self.cam.get_hpr()}")

        # Adjust near/far planes based on bounds
        # fov = self.camLens.get_fov() # Not directly used here but useful for context
        near_plane = max(
            0.01, radius / 100.0
        )  # Avoid zero near plane, use smaller value relative to radius
        far_plane = max(
            100.0, default_distance + radius * 5.0
        )  # Ensure far plane covers model + distance
        self.camLens.set_near_far(near_plane, far_plane)
        print(f"Camera Near/Far planes set to: {near_plane:.3f} / {far_plane:.2f}")

        # --- Lighting Setup ---
        self.render.clear_light()
        self.model_root.clear_light()  # Clear lights attached below model_root if any

        dlight = p3d.DirectionalLight("dlight")
        dlight.setColor((0.2, 0.2, 0.2, 1.6))  # Slightly stronger directional light
        dlnp = self.render.attach_new_node(dlight)
        dlnp.set_hpr(60, -30, 0)  # Angle the light
        self.render.set_light(dlnp)

        self.ambient_light_node = p3d.AmbientLight("ambient_light")
        self.ambient_light_node.set_color(
            (0.3, 0.3, 0.3, 1)
        )  # Slightly brighter ambient
        self.ambient_light_nodepath = self.render.attach_new_node(
            self.ambient_light_node
        )
        self.render.set_light(self.ambient_light_nodepath)

        # --- Animation ---
        char_node = self.model_root.find(
            "**/+Character"
        )  # Find any node with Character component
        if char_node:
            self.anims = p3d.AnimControlCollection()
            p3d.autoBind(
                self.model_root.node(), self.anims, ~0
            )  # Bind animations under model_root
            if self.anims.get_num_anims() > 0:
                anim_name = self.anims.get_anim_name(0)
                self.anims.play(anim_name)
                self.anims.loop(anim_name, True)
                print(f"Playing and looping animation: {anim_name}")
            else:
                print("Character node found, but no animations available.")
        else:
            print("No Character node found, skipping animation setup.")

    def load_model(self):
        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        if not infile.exists():
            print(f"Error: Input file not found at {infile.to_os_specific()}")
            sys.exit(1)

        p3d.get_model_path().prepend_directory(infile.get_dirname())
        print(f"Loading model: {infile.get_basename()}...")
        try:
            self.model_root = self.loader.load_model(infile, noCache=True)
            if self.model_root is None or self.model_root.isEmpty():
                print(
                    f"Error: Failed to load model or model is empty: {infile.to_os_specific()}"
                )
                sys.exit(1)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {infile.to_os_specific()}: {e}")
            sys.exit(1)
        self.model_root.reparent_to(self.render)

    def intialise_shortcuts(self):
        self.accept("escape", sys.exit)
        self.accept("q", sys.exit)
        self.accept("w", self.toggle_wireframe)
        self.accept("t", self.toggle_texture)
        self.accept("n", self.toggle_normal_maps)
        self.accept("e", self.toggle_emission_maps)
        self.accept("o", self.toggle_occlusion_maps)
        self.accept(
            "a", self._generate_new_nodes, [5]
        )  # Bind 'a' to generate 5 nodes internally
        self.accept("shift-l", self.model_root.ls)
        self.accept("shift-a", self.model_root.analyze)
        self.accept("f1", self.take_current_view_screenshot)
        self.accept(
            "f2", self.take_custom_view_screenshot, [(0, -20, 5), (0, -15, 0)]
        )  # Example pos/hpr

    def depthmap_setup(self):
        self.depth_map_size = 512  # Resolution of the depth map
        self.depth_buffer = self.setup_depth_buffer(self.depth_map_size)
        self.depth_texture = p3d.Texture()  # Create texture to hold depth
        self.depth_texture.setFormat(p3d.Texture.T_depth)
        self.depth_buffer.addRenderTexture(
            self.depth_texture,
            GraphicsOutput.RTM_depth,
        )  # Mode to copy depth
        # Create a camera for the depth map rendering
        self.depth_cam_node = Camera("depthCam")
        # Use a lens appropriate for your needs (e.g., perspective matching main cam, or ortho for shadows)
        # lens = OrthographicLens()
        # lens.setFilmSize(20, 20) # Example for ortho
        lens = PerspectiveLens()
        lens.setFilmSize(self.camLens.getFilmSize())  # Match main camera's FOV/aspect
        lens.setFov(self.camLens.getFov())
        lens.setNearFar(
            self.camLens.getNear(), self.camLens.getFar()
        )  # Match near/far planes
        self.depth_cam_node.setLens(lens)
        self.depth_cam = self.render.attachNewNode(self.depth_cam_node)

    def depthmap_position_on_lens(self):
        self.depth_cam.setPos(self.cam.getPos(self.render))
        self.depth_cam.setHpr(self.cam.getHpr(self.render))

    def take_current_view_screenshot(self):
        """Takes screenshot of the current camera view and saves it."""
        filepath = f"current_camera_view_{self.random_string(4)}.png"
        print(f"Taking screenshot of current view to: {filepath}")
        # Pass self (the App instance) to the screenshot function
        success = take_screenshot(self, filepath=filepath, return_data=False)
        if success:
            print("Screenshot saved.")
        else:
            print("Screenshot failed.")

    def take_custom_view_screenshot(
        self, pos: tuple[float, float, float], hpr: tuple[float, float, float]
    ):
        """Takes screenshot from a predefined custom view and saves it."""
        filepath = f"custom_camera_view_{self.random_string(4)}.png"
        print(f"Taking screenshot from custom view: Pos={pos}, HPR={hpr} to {filepath}")
        # Pass self (the App instance) to the screenshot function
        success = take_screenshot(
            self, filepath=filepath, cam_pos=pos, cam_hpr=hpr, return_data=False
        )
        if success:
            print("Screenshot saved.")
        else:
            print("Screenshot failed.")

    # --- Toggles ---
    def toggle_normal_maps(self):
        self.pipeline.use_normal_maps = not self.pipeline.use_normal_maps
        print(
            f"Normal maps toggled: {'ON' if self.pipeline.use_normal_maps else 'OFF'}"
        )

    def toggle_emission_maps(self):
        self.pipeline.use_emission_maps = not self.pipeline.use_emission_maps
        print(
            f"Emission maps toggled: {'ON' if self.pipeline.use_emission_maps else 'OFF'}"
        )

    def toggle_occlusion_maps(self):
        self.pipeline.use_occlusion_maps = not self.pipeline.use_occlusion_maps
        print(
            f"Occlusion maps toggled: {'ON' if self.pipeline.use_occlusion_maps else 'OFF'}"
        )

    def toggle_ambient_light(self):
        if self.render.has_light(self.ambient_light_nodepath):
            self.render.clear_light(self.ambient_light_nodepath)
            print("Ambient light OFF")
        else:
            self.render.set_light(self.ambient_light_nodepath)
            print("Ambient light ON")

    # --- Utility ---
    def random_string(self, length=8):
        """Generates a random alphanumeric string."""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    # --- Core Simulation Logic ---

    def _generate_new_nodes(self, sample_size: int = 1) -> None:
        """
        Generates new camera nodes, takes screenshots, and updates the internal full_graph state.
        This method modifies self.full_graph directly.
        Seeding (random, numpy) should be done *before* calling this if reproducibility is needed.
        """
        print(f"\n--- Generating {sample_size} Camera Nodes for Graph ---")

        # Get tight bounds for more accurate positioning relative to visible geometry
        min_pt, max_pt = Point3(), Point3()
        self.model_root.calc_tight_bounds(
            min_pt, max_pt, self.render
        )  # Calculate in world space

        if min_pt == max_pt:
            print(
                "Warning: Could not get tight bounds for node generation. Using potentially less accurate bounds."
            )
            # Fallback to regular bounds if tight bounds fail
            bounds = self.model_root.getBounds()
            if bounds.is_empty():
                print("Error: Cannot generate nodes, model bounds are empty.")
                return
            center = bounds.get_center()
            radius = bounds.get_radius()
            min_pt = center - Vec3(radius, radius, radius)
            max_pt = center + Vec3(radius, radius, radius)
            if radius < 0.1:
                radius = 0.5  # Clamp small radius
        else:
            center = (min_pt + max_pt) / 2.0
            radius = (max_pt - center).length()
            if radius < 0.1:
                radius = 0.5  # Clamp small radius

        min_x, min_y, min_z = min_pt.x, min_pt.y, min_pt.z
        max_x, max_y, max_z = max_pt.x, max_pt.y, max_pt.z

        print(
            f"Using Bounds for Generation: Min({min_x:.2f}, {min_y:.2f}, {min_z:.2f}), Max({max_x:.2f}, {max_y:.2f}, {max_z:.2f})"
        )
        print(
            f"Center: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f}), Radius: {radius:.2f}"
        )

        # Handle cases where bounds might be planar or linear
        span_x = max(0.1, max_x - min_x)  # Ensure minimum span
        span_y = max(0.1, max_y - min_y)
        span_z = max(0.1, max_z - min_z)
        max_span = max(span_x, span_y)  # Max horizontal span

        # Define generation parameters dynamically based on bounds
        # Adjust distance based on model size
        dist_factor = 2.0  # Base distance factor from center
        dist_min = (
            radius * (dist_factor * 0.8) + max_span * 0.5
        )  # Closer for smaller span
        dist_max = (
            radius * (dist_factor * 1.2) + max_span * 1.0
        )  # Further for larger span
        dist_min = max(1.0, dist_min)  # Ensure minimum distance
        dist_max = max(dist_min + 1.0, dist_max)  # Ensure max > min

        # Z offset relative to the center
        z_offset_min = span_z * 0.5  # Look from slightly above center
        z_offset_max = (
            span_z * 1.0 + radius * 0.5
        )  # Look from higher up, relative to radius
        z_offset_min = max(0.5, z_offset_min)  # Min height offset
        z_offset_max = max(z_offset_min + 0.5, z_offset_max)  # Ensure max > min

        # Look-at variation around the center
        look_variation_factor = 0.15  # Percentage of span/radius
        look_var_x = span_x * look_variation_factor
        look_var_y = span_y * look_variation_factor
        look_var_z = span_z * look_variation_factor

        newly_added_nodes = []  # Keep track of nodes added in this call

        for i in range(sample_size):
            # --- Generate Position (Spherical coordinates around center) ---
            distance = random.uniform(dist_min, dist_max)
            theta = random.uniform(0, 2 * math.pi)  # Angle in XY plane (0 to 360)
            phi_angle_deg = random.uniform(
                15, 75
            )  # Angle from Z+ axis (15=high, 75=lower angle)
            phi = math.radians(phi_angle_deg)

            # Convert spherical to Cartesian offsets from center
            offset_x = distance * math.sin(phi) * math.cos(theta)
            offset_y = distance * math.sin(phi) * math.sin(theta)
            offset_z = distance * math.cos(phi)

            # Calculate final camera position
            cam_x = center.x + offset_x
            cam_y = center.y + offset_y
            cam_z = center.z + max(
                z_offset_min, offset_z
            )  # Ensure camera is reasonably high
            pos = (cam_x, cam_y, cam_z)

            # --- Generate Rotation (LookAt) ---
            # Target point slightly randomized around the model center
            target_x = center.x + random.uniform(-look_var_x, look_var_x)
            target_y = center.y + random.uniform(-look_var_y, look_var_y)
            target_z = center.z + random.uniform(-look_var_z, look_var_z)
            look_at_target = p3d.Point3(target_x, target_y, target_z)

            # Use a temporary node to calculate HPR easily
            temp_cam_node = NodePath(f"temp_cam_for_hpr_{count}")
            temp_cam_node.set_pos(pos)
            temp_cam_node.look_at(look_at_target)
            rot = temp_cam_node.get_hpr()
            temp_cam_node.remove_node()  # Clean up temporary node

            print(
                f"Node {count}: Pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), HPR({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}) -> Target({target_x:.2f}, {target_y:.2f}, {target_z:.2f})"
            )

            image_data = take_screenshot(
                self, cam_pos=pos, cam_hpr=rot, return_data=True
            )

            if image_data is None or image_data.is_null():
                print(
                    f"Warning: Failed to get valid screenshot data for node {count}. Skipping node."
                )
                continue  # Skip adding this node if screenshot failed

            # --- Create Node State ---
            # Get positions/rotations of *existing* nodes *before* adding the new one
            position_of_other_nodes: Dict[int, Tuple[float, float, float]] = {}
            rotation_of_other_nodes: Dict[int, Tuple[float, float, float]] = {}
            for node_id, existing_node in full_graph.items():
                position_of_other_nodes[node_id] = existing_node.position
                rotation_of_other_nodes[node_id] = existing_node.rotation

            new_node_state = SingleNodeState(
                position=pos,
                rotation=rot,
                depth_map=image_data,  # Assuming PNMImage is the desired format here
                overlap_map={},  # Placeholder for overlap data
                position_of_other_nodes=position_of_other_nodes,
                rotation_of_other_nodes=rotation_of_other_nodes,
                id=count,
            )

            # --- Update Graph State ---
            # Add the new node to the main graph
            full_graph[count] = new_node_state
            newly_added_nodes.append((count, pos, rot))

            # Update existing nodes with info about the *new* node
            for node_id in (
                position_of_other_nodes.keys()
            ):  # Iterate through IDs that existed *before* this loop iteration
                # Check if the node still exists (might be relevant in complex scenarios)
                if node_id in full_graph:
                    full_graph[node_id].position_of_other_nodes[count] = pos
                    full_graph[node_id].rotation_of_other_nodes[count] = rot

            count += 1  # Increment global count *after* successfully adding node

        print(
            f"--- Camera Node Generation Complete ({len(newly_added_nodes)} nodes added) ---"
        )
        # This method modifies self.full_graph directly, so no explicit return of graph needed.


def reset(seed: Optional[int] = None) -> tuple[FullGraphState, AllBarcodeClassifiers]:
    print(f"\n--- Resetting Environment with Seed: {seed} ---")

    # 1. Set Seeds
    if seed is None:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed if none provided
        print(f"Generated random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)  # Seed NumPy's random generator

    # 2. Clear State
    full_graph = {}
    all_actions = {}
    count = 0

    # 3. Generate Initial State (e.g., generate 1 starting node)
    # Seeding is already done, so node generation will be deterministic for a given seed

    sample_size = INIT_SAMPLE_SIZE
    # 4. Prepare Initial Observations/Classifiers (placeholder)
    # This might involve analyzing the initial graph state if needed
    initial_classifiers: AllBarcodeClassifiers = {}  # Placeholder

    print("--- Environment Reset Complete ---")
    return full_graph, initial_classifiers


def step(actions: AllActions) -> tuple[FullGraphState, AllBarcodeClassifiers]:
    print(f"\n--- Executing Step with Actions: {actions} ---")
    all_actions = actions  # Store the received actions

    delta_pos: tuple[float, float, float] = (0, 0, 0)
    delta_rot: tuple[float, float, float] = (0, 0, 0)
    for i, sa in all_actions.items():
        delta_pos = cast(
            tuple[float, float, float],
            tuple(x + y for x, y in zip(delta_pos, sa.delta_pos)),
        )
        delta_rot = cast(
            tuple[float, float, float],
            tuple(x + y for x, y in zip(delta_rot, sa.delta_rot)),
        )

    current_classifiers: AllBarcodeClassifiers = {}

    for i, sns in full_graph.items():
        sns.position = cast(
            tuple[float, float, float],
            tuple(x + y for x, y in zip(sns.position, delta_pos)),
        )
        sns.rotation = cast(
            tuple[float, float, float],
            tuple(x + y for x, y in zip(sns.rotation, delta_rot)),
        )
        current_classifiers[i] = SingleBarcodeClassifier(
            (sns.depth_map.width, sns.depth_map.height), sns.depth_map, i
        )

    print("--- Step Execution Complete ---")
    # Return the *current* state of the graph after actions
    return full_graph, current_classifiers


def main():
    app = App()

    initial_seed = 123
    initial_graph_state, initial_classifiers = reset(seed=initial_seed)
    print(f"Initial Graph State Nodes: {len(initial_graph_state)}")

    app.run()


if __name__ == "__main__":
    main()
