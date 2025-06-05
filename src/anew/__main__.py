import sys
import os
import random
from pathlib import Path
import math

# --- Panda3D Imports ---
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Filename,
    Texture,
    Point3,
    Camera,
    NodePath,
    AmbientLight,
    DirectionalLight,
    TexGenAttrib,
    TextureStage,
    Vec4,
    Point2,
    OrthographicLens,
)

# --- Import DirectGUI elements ---
from direct.gui.DirectGui import DirectFrame

from generators import generate_random_box, select_decal_texture

# --- Configuration ---
SCRIPT_DIR = Path.cwd()  # Get script directory
TEXTURE_DIR = SCRIPT_DIR / "textures"  # Path to textures folder
DECAL_DIR = SCRIPT_DIR / "decals"  # Path to textures folder
NUM_BOXES = 1  # Keep at 1 for this setup
# EPSILON is no longer needed for the overlay


# --- Helper Function to Create a 3D Box (Unchanged) ---
# --- Main Application Class ---
class RandomBoxesApp(ShowBase):
    def __init__(self, seed):
        # Basic ShowBase setup
        super().__init__()
        self.seed = seed
        self.setBackgroundColor(0.1, 0.1, 0.15)  # Dark background
        self.disableMouse()  # Prevent mouse interference

        # Load all textures from the texture directory
        self.textures = self.load_textures(TEXTURE_DIR)
        if not self.textures:
            print(f"No textures found in '{TEXTURE_DIR}'!")
            sys.exit(1)

        # Setup lighting
        self.setup_lighting()

        self.projector_viz_enabled = True
        # --- Generate the single box AT THE ORIGIN ---
        self.boxes = []
        if NUM_BOXES > 0:
            box_np = generate_random_box(self.seed, 0, self.textures, self.render)
            self.boxes.append(box_np)
            self.target_object = box_np
        else:
            print("NUM_BOXES is set to 0. No box created.")
            self.target_object = None
            sys.exit()

        # --- Position the camera RANDOMLY looking at the origin ---
        self.set_random_camera_view()

        # --- Create the 2D OVERLAY rectangle ---
        # This needs to happen AFTER the camera is positioned, as projection depends on view
        self.overlay_frame = self.create_overlay_rectangle()

        if self.target_object:
            self.setup_projector(select_decal_texture(DECAL_DIR))
        else:
            print("Skipping projector setup because target object is missing.")

        self.render.set_shader_auto()

        # Basic Controls
        self.accept("escape", sys.exit)
        self.accept("space", self.regenerate_view_and_overlay)
        self.accept("p", self.toggle_projector_visualization)
        print("Press SPACE to change view and overlay position.")
        print("Press ESC to exit.")

    def create_overlay_rectangle(self):
        """Creates a white 2D rectangle positioned relative to the projected box center."""
        # --- Body copied from previous version ---
        # 1. Define the 3D point to project (the box center)
        box_center_3d = Point3(0, 0, 0)

        # 2. Project the 3D point using the MAIN camera's lens
        projected_2d = Point2()
        # Check if the point projects onto the screen (is not behind the camera)
        # Use self.camNode which is the Camera node for the main camera
        if not self.camNode.getLens().project(
            self.render.getRelativePoint(self.camera, box_center_3d), projected_2d
        ):
            # ^ Project point relative to the camera for correct results
            print(
                "Warning: Box center is behind the main camera, cannot create overlay."
            )
            return None  # Cannot draw overlay if center isn't visible

        # 3. Convert clip space (-1 to 1) to aspect2d coordinates
        aspect_ratio = self.getAspectRatio()
        screen_center_x = projected_2d.x * aspect_ratio
        screen_center_y = projected_2d.y

        # 4. Define random size and offset for the 2D overlay IN aspect2d UNITS
        overlay_width = random.uniform(0.1, 0.4)
        overlay_height = random.uniform(0.1, 0.4)
        offset_x = random.uniform(-0.3, 0.3)
        offset_y = random.uniform(-0.3, 0.3)

        # Calculate final screen position (center of the overlay frame)
        final_x = screen_center_x + offset_x
        final_y = screen_center_y + offset_y

        # 5. Create the DirectFrame (the white rectangle)
        overlay = DirectFrame(
            parent=aspect2d,  # Parent to the 2D space corrected for aspect ratio
            frameColor=(1, 1, 1, 1),  # Force white color
            frameSize=(
                final_x - overlay_width / 2.0,
                final_x + overlay_width / 2.0,  # Left, Right
                final_y - overlay_height / 2.0,
                final_y + overlay_height / 2.0,  # Bottom, Top
            ),
            suppressMouse=True,  # Prevent mouse interaction
        )
        print(
            f"Created overlay relative to perspective view at screen coords: ({final_x:.2f}, {final_y:.2f})"
        )
        return overlay
        # --- End of copied body ---

    def regenerate_view_and_overlay(self):
        """Repositions camera and overlay"""
        print("Regenerating view...")
        # Reposition camera
        self.set_random_camera_view()
        # Remove old overlay if it exists
        if hasattr(self, "overlay_frame") and self.overlay_frame:
            self.overlay_frame.destroy()
        # Create new overlay based on new camera view
        self.overlay_frame = self.create_overlay_rectangle()

    def load_textures(self, folder_path):
        """Loads textures from a given folder path."""
        textures = []
        print(f"Loading textures from: {folder_path}")
        try:
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tga")):
                    fpath = folder_path / fname
                    tex = self.loader.loadTexture(Filename.from_os_specific(str(fpath)))
                    if tex:
                        print(f"  Loaded: {fname}")
                        tex.set_wrap_u(Texture.WM_repeat)
                        tex.set_wrap_v(Texture.WM_repeat)
                        tex.set_anisotropic_degree(4)
                        textures.append(tex)
                    else:
                        print(f"  Warning: Failed to load {fname}")
        except FileNotFoundError:
            print(f"  Error: Folder not found at {folder_path}")
        except Exception as e:
            print(f"  Error loading textures: {e}")
        return textures

    def setup_lighting(self):
        """Sets up basic ambient and directional lighting."""
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color((0.4, 0.4, 0.4, 1))  # Slightly brighter ambient
        self.ambient_light_np = self.render.attach_new_node(ambient_light)
        self.render.set_light(self.ambient_light_np)
        directional_light = DirectionalLight("directional_light")
        directional_light.set_color((0.8, 0.8, 0.75, 1))
        self.directional_light_np = self.render.attach_new_node(directional_light)
        # Randomize light direction slightly too each time? Optional.
        self.directional_light_np.set_hpr(
            random.uniform(0, 360), random.uniform(-70, -20), 0
        )
        self.render.set_light(self.directional_light_np)

    def set_random_camera_view(self):
        """Sets a random camera position looking at the origin."""
        # Determine random distance and angles for camera positioning
        distance = random.uniform(8, 15)  # How far from origin
        # Angle in XY plane (0=along +X, pi/2=along +Y, pi=along -X, 3pi/2=along -Y)
        theta = random.uniform(0, 2 * math.pi)
        # Angle from Z+ axis (0=straight above, pi/2=level with origin, pi=straight below)
        # Limit phi to avoid being too close to straight up/down for better views
        phi = random.uniform(
            math.pi / 4, 3 * math.pi / 4
        )  # Approx 45 to 135 degrees from Z+

        # Convert spherical coordinates to Cartesian coordinates
        cam_x = distance * math.sin(phi) * math.cos(theta)
        cam_y = distance * math.sin(phi) * math.sin(theta)
        cam_z = distance * math.cos(phi)

        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.lookAt(0, 0, 0)  # Always look at the origin where the box is
        print(f"Camera Pos: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) Looking at origin.")

    def setup_projector(self, texture):
        """Sets up the projector camera and applies projection to the target object."""
        print("Setting up texture projector...")

        if not self.target_object or self.target_object.is_empty():
            print("ERROR: Cannot set up projector without a valid target object.")
            return

        # 1. Create Projector Camera Node and NodePath
        proj_cam_node = Camera(f"projectorCamNode_{texture}")
        self.projectorNP = NodePath(proj_cam_node)

        # 2. Setup Projector Lens
        ortho_lens = OrthographicLens()
        film_width = 1
        film_height = 1

        ortho_lens.setFilmSize(film_width, film_height)
        ortho_lens.setNearFar(1, 100)

        # Consider setting aspect ratio based on PROJECTOR_TEXTURE_FILE dimensions later
        proj_cam_node.setLens(ortho_lens)

        # 3. Position the Projector Camera in the World
        proj_pos = Point3(
            random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(5, 15)
        )  # Random position
        # Ensure projector is not exactly at origin if box is there
        while proj_pos.length_squared() < 1.0:
            proj_pos = Point3(
                random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(5, 15)
            )

        self.projectorNP.setPos(proj_pos)
        self.projectorNP.lookAt(0, 0, 0)  # Point at the box origin
        self.projectorNP.reparentTo(self.render)
        print(f"Projector Camera Pos: {self.projectorNP.getPos(self.render)}")

        # 4. Load the Texture to Project
        try:
            self.projTexture = self.loader.loadTexture(
                Filename.from_os_specific(str(texture))
            )
            if not self.projTexture:
                raise ValueError("Texture failed to load")

            self.projTexture.setWrapU(Texture.WM_border_color)
            self.projTexture.setWrapV(Texture.WM_border_color)
            self.projTexture.setBorderColor(
                Vec4(0, 0, 0, 0)
            )  # Transparent black border
            print(f"Loaded projector texture: {texture}")

        except Exception as e:
            print(f"ERROR: Could not load projector texture '{texture}': {e}")
            self.projectorNP.removeNode()
            self.projectorNP = None
            return  # Stop setup if texture fails

        # 5. Create a Texture Stage for the Projection
        self.projTS = TextureStage("projectorTS")
        self.projTS.setMode(TextureStage.MReplace)  # Or MModulate, MAdd, etc.
        self.projTS.setSort(10)  # Apply after default stage (sort 0)

        # 6. Apply Projection to the Target Object
        # Ensure target object is valid
        if not self.target_object or self.target_object.is_empty():
            print("ERROR: Target object invalid, cannot apply projection.")
            # Clean up projector resources?
            self.projectorNP.removeNode()
            if hasattr(self, "projTexture") and self.projTexture:
                self.projTexture.releaseAll()
            return

        self.target_object.setTexture(self.projTS, self.projTexture)
        self.target_object.setTexProjector(self.projTS, self.projectorNP, self.render)

        self.target_object.setTexGen(self.projTS, TexGenAttrib.MWorldPosition)
        # 7. (Optional) Projector Visualization Setup (initially hidden)
        self.projector_viz = self.loader.loadModel("misc/camera")
        if self.projector_viz and not self.projector_viz.isEmpty():
            self.projector_viz.setScale(0.5)
            self.projector_viz.reparentTo(self.projectorNP)
            self.projector_viz.hide()  # Hide the camera model itself

            # Hide the frustum visualization using the Camera node's method
            print("DEBUG: Hiding frustum initially.")
            proj_cam_node.hide_frustum()  # <<< CORRECTED: Use hide_frustum() on the Camera node
        else:
            print("Warning: Could not load misc/camera model for projector viz.")
            self.projector_viz = None

    def toggle_projector_visualization(self):
        """Toggles visibility of projector model and frustum."""
        if not hasattr(self, "projectorNP") or not self.projectorNP:
            print("Projector not set up, cannot toggle visualization.")
            return

        # Get the actual Camera node
        proj_cam_node = self.projectorNP.node()
        if not isinstance(proj_cam_node, Camera):
            print(
                f"ERROR: projectorNP does not contain a Camera node (found {type(proj_cam_node)})."
            )
            return

        self.projector_viz_enabled = not self.projector_viz_enabled
        if self.projector_viz_enabled:
            print("Showing projector visualization")
            if self.projector_viz and not self.projector_viz.isEmpty():
                self.projector_viz.show()
            # Show frustum using the Camera node's method
            proj_cam_node.show_frustum()  # <<< CORRECTED

        else:
            print("Hiding projector visualization")
            if self.projector_viz and not self.projector_viz.isEmpty():
                self.projector_viz.hide()
            # Hide frustum using the Camera node's method
            proj_cam_node.hide_frustum()  # <<< CORRECTED


# --- Run the Application ---
if __name__ == "__main__":
    app = RandomBoxesApp(3)
    app.run()
