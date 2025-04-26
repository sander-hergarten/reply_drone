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
    TextureStage,
    load_prc_file_data,
    Vec3,
    Vec4,
    Point3,
    Point2, # Needed for 2D projection
    NodePath,
    PandaNode,
    Material,
    AmbientLight,
    DirectionalLight,
    PerspectiveLens,
    TransformState,
    TransparencyAttrib,
    GeomNode,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    GeomTriangles,
    Geom,
    CullFaceAttrib,
    TextureAttrib,
    # CardMaker is no longer needed for the overlay
)
# --- Import DirectGUI elements ---
from direct.gui.DirectGui import DirectFrame

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent # Get script directory
TEXTURE_DIR = SCRIPT_DIR / "textures"      # Path to textures folder
NUM_BOXES = 1                             # Keep at 1 for this setup
# EPSILON is no longer needed for the overlay

# Check if texture folder exists
if not TEXTURE_DIR.is_dir():
    print(f"ERROR: Texture folder not found: {TEXTURE_DIR}")
    print("Please create it and add some texture images.")
    sys.exit(1)


# --- Helper Function to Create a 3D Box (Unchanged) ---
def create_box(width, height, depth, name="Box"):
    """
    Procedurally creates a textured 3D box geometry. Fixed.
    Panda3D Coords: +X=Right, +Y=Into Screen, +Z=Up
    width=X, height=Y, depth=Z
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    vertices = [
        Point3(-w,-h,-d), Point3( w,-h,-d), Point3( w, h,-d), Point3(-w, h,-d), # Bottom (-z)
        Point3(-w,-h, d), Point3( w,-h, d), Point3( w, h, d), Point3(-w, h, d)  # Top (+z)
    ]
    uvs = [(0,0),(1,0),(1,1),(0,1)]
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData(name+'Data', vformat, Geom.UH_static)
    vdata.set_num_rows(24)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    texcoord = GeomVertexWriter(vdata, 'texcoord')
    tris = GeomTriangles(Geom.UH_static)
    def add_face(v_indices, norm_vec):
        start = vertex.get_write_row()
        vertex.add_data3(vertices[v_indices[0]])
        vertex.add_data3(vertices[v_indices[1]])
        vertex.add_data3(vertices[v_indices[2]])
        vertex.add_data3(vertices[v_indices[3]])
        for _ in range(4): normal.add_data3(norm_vec); texcoord.add_data2(uvs[_])
        tris.add_vertices(start+0,start+1,start+2); tris.add_vertices(start+2,start+3,start+0)
        tris.close_primitive()
    add_face([1,0,4,5],Vec3(0,0,-1)); add_face([3,2,6,7],Vec3(0,0,1))
    add_face([0,3,7,4],Vec3(-1,0,0)); add_face([2,1,5,6],Vec3(1,0,0))
    add_face([0,1,2,3],Vec3(0,-1,0)); add_face([5,4,7,6],Vec3(0,1,0))
    geom = Geom(vdata); geom.add_primitive(tris); node = GeomNode(name+'GeomNode')
    node.add_geom(geom); np = NodePath(node)
    np.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
    return np

# --- Main Application Class ---
class RandomBoxesApp(ShowBase):
    def __init__(self):
        # Basic ShowBase setup
        super().__init__()
        self.setBackgroundColor(0.1, 0.1, 0.15) # Dark background
        self.disableMouse() # Prevent mouse interference

        # Load all textures from the texture directory
        self.textures = self.load_textures(TEXTURE_DIR)
        if not self.textures:
            print(f"No textures found in '{TEXTURE_DIR}'!")
            sys.exit(1)

        # Setup lighting
        self.setup_lighting()

        # --- Generate the single box AT THE ORIGIN ---
        self.boxes = []
        if NUM_BOXES > 0:
            box_np = self.create_random_box(0)
            self.boxes.append(box_np)
        else:
             print("NUM_BOXES is set to 0. No box created.")
             sys.exit()

        # --- Position the camera RANDOMLY looking at the origin ---
        self.set_random_camera_view()

        # --- Create the 2D OVERLAY rectangle ---
        # This needs to happen AFTER the camera is positioned, as projection depends on view
        self.overlay_frame = self.create_overlay_rectangle()

        # Enable shader generation for lighting
        self.render.set_shader_auto()

        # Basic Controls
        self.accept("escape", sys.exit)
        # Add a key to regenerate view and overlay
        self.accept("space", self.regenerate_view_and_overlay)
        print("Press SPACE to change view and overlay position.")
        print("Press ESC to exit.")

    def regenerate_view_and_overlay(self):
        """Repositions camera and overlay"""
        print("Regenerating view...")
        # Reposition camera
        self.set_random_camera_view()
        # Remove old overlay if it exists
        if hasattr(self, 'overlay_frame') and self.overlay_frame:
            self.overlay_frame.destroy()
        # Create new overlay based on new camera view
        self.overlay_frame = self.create_overlay_rectangle()

    def load_textures(self, folder_path):
        """Loads textures from a given folder path."""
        textures = []
        print(f"Loading textures from: {folder_path}")
        try:
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tga')):
                    fpath = folder_path / fname
                    tex = self.loader.loadTexture(Filename.from_os_specific(str(fpath)))
                    if tex:
                        print(f"  Loaded: {fname}")
                        tex.set_wrap_u(Texture.WM_repeat)
                        tex.set_wrap_v(Texture.WM_repeat)
                        tex.set_anisotropic_degree(4)
                        textures.append(tex)
                    else: print(f"  Warning: Failed to load {fname}")
        except FileNotFoundError: print(f"  Error: Folder not found at {folder_path}")
        except Exception as e: print(f"  Error loading textures: {e}")
        return textures

    def setup_lighting(self):
        """Sets up basic ambient and directional lighting."""
        ambient_light = AmbientLight('ambient_light')
        ambient_light.set_color((0.4, 0.4, 0.4, 1)) # Slightly brighter ambient
        self.ambient_light_np = self.render.attach_new_node(ambient_light)
        self.render.set_light(self.ambient_light_np)
        directional_light = DirectionalLight('directional_light')
        directional_light.set_color((0.8, 0.8, 0.75, 1))
        self.directional_light_np = self.render.attach_new_node(directional_light)
        # Randomize light direction slightly too each time? Optional.
        self.directional_light_np.set_hpr(random.uniform(0,360), random.uniform(-70,-20), 0)
        self.render.set_light(self.directional_light_np)

    def create_random_box(self, index):
        """Creates a single 3D box, fixed at origin, with random texture/rotation."""
        # Random size for the main box
        main_box_width = random.uniform(2.0, 5.0)
        main_box_height = random.uniform(2.0, 5.0)
        main_box_depth = random.uniform(2.0, 5.0)

        # Create main box geometry
        main_box_np = create_box(main_box_width, main_box_height, main_box_depth, name=f"MainBox_{index}")

        # --- Position the box at the ORIGIN ---
        main_box_np.setPos(0, 0, 0)

        # Random rotation IS STILL APPLIED
        main_box_np.setHpr(random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))

        # Apply random texture
        if self.textures:
            tex = random.choice(self.textures)
            main_box_np.setTexture(tex)
        else:
            main_box_np.setColor(random.random(), random.random(), random.random(), 1)

        # --- NO Patch is created on the 3D surface anymore ---

        # Attach the main box to the scene graph
        main_box_np.reparentTo(self.render)
        return main_box_np

    def set_random_camera_view(self):
        """Sets a random camera position looking at the origin."""
        # Determine random distance and angles for camera positioning
        distance = random.uniform(8, 15) # How far from origin
        # Angle in XY plane (0=along +X, pi/2=along +Y, pi=along -X, 3pi/2=along -Y)
        theta = random.uniform(0, 2 * math.pi)
        # Angle from Z+ axis (0=straight above, pi/2=level with origin, pi=straight below)
        # Limit phi to avoid being too close to straight up/down for better views
        phi = random.uniform(math.pi / 4, 3 * math.pi / 4) # Approx 45 to 135 degrees from Z+

        # Convert spherical coordinates to Cartesian coordinates
        cam_x = distance * math.sin(phi) * math.cos(theta)
        cam_y = distance * math.sin(phi) * math.sin(theta)
        cam_z = distance * math.cos(phi)

        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.lookAt(0, 0, 0) # Always look at the origin where the box is
        print(f"Camera Pos: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) Looking at origin.")


    def create_overlay_rectangle(self):
        """Creates a white 2D rectangle positioned relative to the projected box center."""
        # 1. Define the 3D point to project (the box center)
        box_center_3d = Point3(0, 0, 0)

        # 2. Project the 3D point to 2D screen space (clip space -1 to 1)
        projected_2d = Point2()
        # Check if the point projects onto the screen (is not behind the camera)
        if not self.camLens.project(box_center_3d, projected_2d):
            print("Warning: Box center is behind the camera, cannot create overlay.")
            return None # Cannot draw overlay if center isn't visible

        # 3. Convert clip space (-1 to 1) to aspect2d coordinates
        # aspect2d X range is [-aspectRatio, aspectRatio], Y range is [-1, 1]
        aspect_ratio = self.getAspectRatio()
        screen_center_x = projected_2d.x * aspect_ratio
        screen_center_y = projected_2d.y

        # 4. Define random size and offset for the 2D overlay IN aspect2d UNITS
        # Make size somewhat proportional to distance? Maybe not needed for this effect.
        overlay_width = random.uniform(0.1, 0.4)
        overlay_height = random.uniform(0.1, 0.4)
        # Define random offset from the projected center
        offset_x = random.uniform(-0.3, 0.3)
        offset_y = random.uniform(-0.3, 0.3)

        # Calculate final screen position (center of the overlay frame)
        final_x = screen_center_x + offset_x
        final_y = screen_center_y + offset_y

        # 5. Create the DirectFrame (the white rectangle)
        # frameSize defines (left, right, bottom, top) edges in parent's coordinate system
        overlay = DirectFrame(
            parent=aspect2d, # Parent to the 2D space corrected for aspect ratio
            frameColor=(1, 1, 1, 1), # Force white color
            frameSize=(
                final_x - overlay_width / 2.0, final_x + overlay_width / 2.0, # Left, Right
                final_y - overlay_height / 2.0, final_y + overlay_height / 2.0, # Bottom, Top
             ),
            suppressMouse = True, # Prevent mouse interaction
        )
        print(f"Created overlay at screen coords: ({final_x:.2f}, {final_y:.2f})")
        return overlay


# --- Run the Application ---
if __name__ == "__main__":
    app = RandomBoxesApp()
    app.run()
