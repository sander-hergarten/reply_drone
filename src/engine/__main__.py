import math
import os
import sys
import random
import string
from dataclasses import dataclass # Make sure dataclasses is imported
from typing import List, Tuple # Import typing helpers

from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomVertexReader
from panda3d.core import NodePath # Import NodePath for type hinting if needed

import panda3d.core as p3d

import simplepbr

# --- Define the Engine dataclass (assuming it's not in engine.api) ---
# If engine.api exists and contains Engine, remove this definition
# and ensure 'from engine.api import Engine' works.
@dataclass
class Engine:
    """
    Stores lists of camera node positions and rotations (HPR).
    """
    node_positions: List[Tuple[float, float, float]]
    node_rotations: List[Tuple[float, float, float]] # Panda3D HPR: Heading, Pitch, Roll
# --- End of Engine Definition ---


p3d.load_prc_file_data(
    '',
    'window-size 1024 768\n'
    'texture-minfilter mipmap\n'
    'texture-anisotropic-degree 16\n'
    # Optional: Add offscreen buffer if screenshots fail without visible window
    # 'window-type offscreen\n'
)

# Global base reference for take_screenshot
base = None

def take_screenshot(filepath, cam_pos=None, cam_hpr=None):
    """Capture screenshot with specific camera position/rotation"""
    global base
    if not base:
        print("Error: ShowBase instance 'base' not available for screenshot.")
        return
    if not base.win:
        print("Error: No window/buffer available to save screenshot from.")
        return

    original_pos = None
    original_hpr = None

    if cam_pos is not None or cam_hpr is not None:
        # Store original camera transform
        original_pos = base.cam.get_pos()
        original_hpr = base.cam.get_hpr()

        # Apply new transform
        if cam_pos: base.cam.set_pos(cam_pos)
        if cam_hpr: base.cam.set_hpr(cam_hpr)

        # Force render updates - crucial for applying changes before saving
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame() # Render twice to be sure

    # Capture screenshot
    success = base.win.saveScreenshot(Filename(filepath))
    if not success:
        print(f"Warning: Failed to save screenshot to {filepath}")


    # Restore original transform if it was changed
    if original_pos is not None or original_hpr is not None:
        if original_pos: base.cam.set_pos(original_pos)
        if original_hpr: base.cam.set_hpr(original_hpr)
        # Render again after restoring if needed, though often not critical
        base.graphicsEngine.renderFrame()


class App(ShowBase):
    def __init__(self):
        global base # Make base accessible globally
        if len(sys.argv) < 2:
            print("Missing input file")
            print("Usage: python your_script_name.py path/to/model.glb")
            sys.exit(1)

        super().__init__()
        base = self # Assign the instance to the global base

        self.pipeline = simplepbr.init()

        # load mesh
        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        if not infile.exists():
             print(f"Error: Input file not found at {infile.to_os_specific()}")
             sys.exit(1)

        p3d.get_model_path().prepend_directory(infile.get_dirname())
        print(f"Loading model: {infile.get_basename()}...")
        try:
            self.model_root = self.loader.load_model(infile, noCache=True)
            if self.model_root is None or self.model_root.isEmpty():
                print(f"Error: Failed to load model or model is empty: {infile.to_os_specific()}")
                sys.exit(1)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {infile.to_os_specific()}: {e}")
            sys.exit(1)


        # shortcuts
        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
        self.accept('w', self.toggle_wireframe)
        self.accept('t', self.toggle_texture)
        self.accept('n', self.toggle_normal_maps)
        self.accept('e', self.toggle_emission_maps)
        self.accept('o', self.toggle_occlusion_maps)
        # self.accept('y', self.toggle_ambient) # Keep Y if you still want ambient toggling
        self.accept('a', self.generate_and_screenshot_nodes, [5]) # Bind 'a' to generate 5 nodes
        self.accept('shift-l', self.model_root.ls)
        self.accept('shift-a', self.model_root.analyze)
        self.accept("f1", self.take_current_view_screenshot) # Use a method for clarity
        self.accept("f2", self.take_custom_view_screenshot) # Use a method

        self.model_root.reparent_to(self.render)

        # --- Camera Setup based on Bounds ---
        print("Calculating model bounds for camera setup...")
        bounds = self.model_root.getBounds()
        center = bounds.get_center()
        radius = 1.0 # Default radius
        if bounds.is_empty():
            print("Warning: Model bounds are empty. Using default camera position.")
            center = p3d.Point3(0, 0, 0)
        else:
            radius = bounds.get_radius()
            print(f"Model Center: {center.getX():.2f}, {center.getY():.2f}, {center.getZ():.2f}")
            print(f"Model Radius: {radius:.2f}")


        # Disable default mouse control if generating static views mainly
        self.disableMouse()

        # Sensible default camera position looking at the center
        # Adjust distance based on radius
        default_distance = radius * 2.5 # Adjust multiplier as needed
        self.cam.set_pos(center + p3d.Vec3(0, -default_distance, radius * 0.5)) # Position back and slightly up
        self.cam.look_at(center) # Make it look at the center
        print(f"Default Camera Pos: {self.cam.get_pos()}, HPR: {self.cam.get_hpr()}")

        # Adjust near/far planes based on bounds
        fov = self.camLens.get_fov()
        # Calculate distance based on FOV if needed, but direct positioning might be simpler
        # distance = radius / math.tan(math.radians(min(fov[0], fov[1]) / 2.0))
        near_plane = max(0.1, radius / 100.0) # Avoid zero near plane
        far_plane = max(1000.0, default_distance + radius * 5.0) # Ensure far plane covers model + distance
        self.camLens.set_near_far(near_plane, far_plane)
        print(f"Camera Near/Far planes set to: {near_plane:.2f} / {far_plane:.2f}")

        # --- Lighting Setup ---
        # Clear existing lights first if necessary
        self.render.clear_light()
        self.model_root.clear_light() # Clear lights attached below model_root

        # Add a key directional light
        dlight = p3d.DirectionalLight('dlight')
        dlight.setColor((0.2, 0.2, 0.2, 1.2))
        dlnp = self.render.attach_new_node(dlight)
        dlnp.set_hpr(60, -30, 0) # Angle the light
        self.render.set_light(dlnp)

        # Add ambient light
        self.ambient_light_node = p3d.AmbientLight('ambient_light')
        self.ambient_light_node.set_color((0.25, 0.25, 0.25, 1)) # Slightly brighter ambient
        self.ambient_light_nodepath = self.render.attach_new_node(self.ambient_light_node)
        self.render.set_light(self.ambient_light_nodepath)

        # If model had lights, optionally re-parent them (consider if needed)
        # for light in self.model_root.find_all_matches('**/+Light'):
        #     light.parent.wrt_reparent_to(self.render)
        #     self.render.set_light(light)

        # --- Animation ---
        if self.model_root.find('**/+Character'):
            self.anims = p3d.AnimControlCollection()
            p3d.autoBind(self.model_root.node(), self.anims, ~0)
            if self.anims.get_num_anims() > 0:
                self.anims.get_anim(0).loop(True)
                print(f"Playing animation: {self.anims.get_anim(0).get_name()}")

        # Print vertices if needed (can be slow for large models)
        # self.print_first_vertices(10)

    def print_first_vertices(self, count=10):
        """Print first N vertices from all geometries"""
        print(f"\nFirst {count} vertices:")
        found = 0
        vertex_data_list = []

        # Using find_all_matches for robustness
        for geom_node_path in self.model_root.find_all_matches('**/+GeomNode'):
            geom_node = geom_node_path.node()
            for i in range(geom_node.get_num_geoms()):
                geom = geom_node.get_geom(i)
                vdata = geom.get_vertex_data()
                vertex_data_list.append(vdata) # Collect unique vdata objects

        # Process unique vertex data objects
        processed_vdata = set()
        for vdata in vertex_data_list:
            if vdata in processed_vdata:
                continue # Skip if already processed
            processed_vdata.add(vdata)

            if not vdata.hasColumn('vertex'):
                continue # Skip if no vertex column

            vertex_reader = GeomVertexReader(vdata, 'vertex')
            while not vertex_reader.isAtEnd() and found < count:
                v = vertex_reader.getData3()
                print(f"Vertex {found + 1}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
                found += 1
            if found >= count:
                break
        if found == 0:
            print("No vertices found or 'vertex' column missing.")


    def take_current_view_screenshot(self):
        """Takes screenshot of the current camera view."""
        filepath = "current_camera_view.png"
        print(f"Taking screenshot of current view: {filepath}")
        # No need to pass pos/hpr, it uses the current camera state
        take_screenshot(filepath)
        print("Screenshot saved.")

    def take_custom_view_screenshot(self):
        """Takes screenshot from a predefined custom view."""
        filepath = "custom_camera_view.png"
        pos = (20, -20, 3)
        hpr = (45, 0, 0) # Looks like H=45 (yaw right), P=0 (level), R=0
        print(f"Taking screenshot from custom view: {filepath} Pos={pos}, HPR={hpr}")
        take_screenshot(filepath, cam_pos=pos, cam_hpr=hpr)
        print("Screenshot saved.")

    def toggle_normal_maps(self):
        self.pipeline.use_normal_maps = not self.pipeline.use_normal_maps
        print(f"Normal maps toggled: {'ON' if self.pipeline.use_normal_maps else 'OFF'}")

    def toggle_emission_maps(self):
        self.pipeline.use_emission_maps = not self.pipeline.use_emission_maps
        print(f"Emission maps toggled: {'ON' if self.pipeline.use_emission_maps else 'OFF'}")

    def toggle_occlusion_maps(self):
        self.pipeline.use_occlusion_maps = not self.pipeline.use_occlusion_maps
        print(f"Occlusion maps toggled: {'ON' if self.pipeline.use_occlusion_maps else 'OFF'}")

    def toggle_ambient_light(self):
        # Check using the nodepath we stored
        if self.render.has_light(self.ambient_light_nodepath):
            self.render.clear_light(self.ambient_light_nodepath)
            print("Ambient light OFF")
        else:
            self.render.set_light(self.ambient_light_nodepath)
            print("Ambient light ON")

    def generate_and_screenshot_nodes(self, sample_size: int = 1):
        """
        Wrapper function to call return_engine, triggered by key press.
        """
        print(f"\nKey pressed: Generating {sample_size} nodes and screenshots...")
        engine_result = self.return_engine(sample_size)
        # engine_result contains the data if you need it later
        print(f"Finished generating {len(engine_result.node_positions)} nodes.")

    def random_string(self, length=8):
        """Generates a random alphanumeric string for filenames."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def return_engine(self, sample_size: int = 1) -> Engine:
        """
        Generates camera nodes based on model bounds, prints their info,
        takes screenshots, and returns an Engine object.

        Args:
            sample_size: The number of camera nodes to generate.

        Returns:
            An Engine dataclass instance containing the lists of positions and rotations.
        """
        print(f"\n--- Generating {sample_size} Camera Nodes ---")

        # Get bounds in render space (world coordinates)
        min_pt, max_pt = p3d.Point3(), p3d.Point3()
        if self.model_root is None or self.model_root.isEmpty() or not self.model_root.calcTightBounds(min_pt, max_pt, self.render):
             print("Warning: Could not calculate tight bounds for the model. Using default bounds [-5,5].")
             min_x, min_y, min_z = -5, -5, 0
             max_x, max_y, max_z = 5, 5, 5
        else:
            min_x, min_y, min_z = min_pt
            max_x, max_y, max_z = max_pt
            print(f"Model World Bounds (Tight): Min({min_x:.2f}, {min_y:.2f}, {min_z:.2f}), Max({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")
            # Handle cases where bounds might be a flat plane (or single point)
            if abs(max_x - min_x) < 0.01: max_x += 0.5; min_x -=0.5
            if abs(max_y - min_y) < 0.01: max_y += 0.5; min_y -=0.5
            # Ensure max_z is somewhat above min_z if model is flat
            if abs(max_z - min_z) < 0.01: max_z += 0.1


        node_positions = []
        node_rotations = []

        # Define generation parameters (adjust as needed)
        # How far above the highest point of the model the camera should be
        z_offset_min = (max_z - min_z) * 0.5 + 2.0 # Relative offset + minimum distance
        z_offset_max = (max_z - min_z) * 1.5 + 10.0 # Relative offset + max distance
        # Ensure min offset is not huge if model is tall but thin
        z_offset_min = max(2.0, z_offset_min)
        z_offset_max = max(z_offset_min + 1.0, z_offset_max) # Ensure max > min

        # Pitch range (looking down) - Panda's Pitch: negative is down
        pitch_min = -85.0 # Almost straight down
        pitch_max = -45.0 # More angled view

        # Heading range (Yaw) - User's specific request range:
        # Either between 60 and 120 OR -120 and -60
        heading_range_positive = (60.0, 120.0)
        heading_range_negative = (-120.0, -60.0)
        # **Alternative: Full 360-degree random heading**
        # heading_min = 0.0
        # heading_max = 360.0

        output_dir = "screenshots_generated" # Define an output directory
        os.makedirs(output_dir, exist_ok=True) # Create it if it doesn't exist


        for i in range(sample_size):
            # Generate random position within XY bounds, and Z above the mesh
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            # Place camera significantly above the highest point
            z = random.uniform(max_z + z_offset_min, max_z + z_offset_max)

            # --- Generate random rotation (HPR) ---
            # Heading (Yaw): Using the specific disjoint ranges from user's original code
            if random.random() < 0.5:
                heading = random.uniform(heading_range_negative[0], heading_range_negative[1])
            else:
                heading = random.uniform(heading_range_positive[0], heading_range_positive[1])
            # **Alternative: Full 360 heading**
            # heading = random.uniform(heading_min, heading_max)

            # Pitch: Looking down within the defined range
            pitch = random.uniform(pitch_min, pitch_max)
            # Roll: Keep level
            roll = 0.0

            pos = (x, y, z)
            rot = (heading, pitch, roll) # HPR order

            node_positions.append(pos)
            node_rotations.append(rot)

            # --- Print the generated node info ---
            print(f"\nNode {i+1}/{sample_size}:")
            print(f"  Position (X, Y, Z): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            print(f"  Rotation (H, P, R): ({rot[0]:.2f}, {rot[1]:.2f}, {rot[2]:.2f})")

            # --- Take screenshot for this node ---
            img_filename = f"node_{i+1:03d}_{self.random_string()}.png" # More descriptive name
            filepath = os.path.join(output_dir, img_filename)
            print(f"  Attempting screenshot: {filepath}")
            try:
                 # Call the global take_screenshot function
                 take_screenshot(filepath, cam_pos=pos, cam_hpr=rot)
                 # Check if file exists after attempt (saveScreenshot returns bool)
                 if os.path.exists(filepath):
                     print(f"  Screenshot saved successfully.")
                 else:
                     print(f"  Screenshot saving failed (check warnings/errors).")
            except Exception as e:
                 # Catch potential errors during the call itself
                 print(f"  Error during screenshot process: {e}")


        print(f"--- Camera Node Generation Complete ({sample_size} nodes) ---")
        return Engine(node_positions=node_positions, node_rotations=node_rotations)

def main():
    # Create and run the app
    app = App()
    app.run()

if __name__ == '__main__':
    main()
