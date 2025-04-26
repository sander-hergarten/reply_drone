"""
rifat

this takes a .gltf or .ply file and displays it in a panda3d window

use: 
    python mesh_viewer.py <path to .gltf or .ply file>

"""

from panda3d.core import Filename

import math
import os
import sys

from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3d

import simplepbr


p3d.load_prc_file_data(
    '',
    'window-size 1024 768\n'
    'texture-minfilter mipmap\n'
    'texture-anisotropic-degree 16\n'
)

def take_screenshot(filepath, cam_pos=None, cam_hpr=None):
    """Capture screenshot with specific camera position/rotation"""
    if cam_pos is not None or cam_hpr is not None:
        # Store original camera transform
        original_pos = base.cam.get_pos()
        original_hpr = base.cam.get_hpr()

        # Apply new transform
        if cam_pos: base.cam.set_pos(cam_pos)
        if cam_hpr: base.cam.set_hpr(cam_hpr)

        # Force render updates
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame()

    # Capture screenshot
    base.win.saveScreenshot(Filename(filepath))

    # Restore original transform
    if cam_pos is not None or cam_hpr is not None:
        base.cam.set_pos(original_pos)
        base.cam.set_hpr(original_hpr)
        base.graphicsEngine.renderFrame()

class App(ShowBase):
    def __init__(self):
        if len(sys.argv) < 2:
            print("Missing input file")
            sys.exit(1)

        super().__init__()

        self.pipeline = simplepbr.init()

        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        p3d.get_model_path().prepend_directory(infile.get_dirname())

        # self.skybox = simplepbr.utils.make_skybox(self.env_map.filtered_env_map)
        # self.skybox.reparent_to(self.render)

        self.model_root = self.loader.load_model(infile, noCache=True)

        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
        self.accept('w', self.toggle_wireframe)
        self.accept('t', self.toggle_texture)
        self.accept('n', self.toggle_normal_maps)
        self.accept('e', self.toggle_emission_maps)
        self.accept('o', self.toggle_occlusion_maps)
        self.accept('a', self.toggle_ambient_light)
        self.accept('shift-l', self.model_root.ls)
        self.accept('shift-a', self.model_root.analyze)
        self.accept("f1", take_screenshot, ["current_camera.png"])
        self.accept("f2", take_screenshot, ["custom_camera.png", (20, -20, 3), (45, 0, 0)])

        self.model_root.reparent_to(self.render)

        bounds = self.model_root.getBounds()
        center = bounds.get_center()
        if bounds.is_empty():
            radius = 1
        else:
            radius = bounds.get_radius()

        fov = self.camLens.get_fov()
        distance = radius / math.tan(math.radians(min(fov[0], fov[1]) / 2.0))
        self.camLens.set_near(min(self.camLens.get_default_near(), radius / 2))
        self.camLens.set_far(max(self.camLens.get_default_far(), distance + radius * 2))
        trackball = self.trackball.node()
        trackball.set_origin(center)
        trackball.set_pos(0, distance, 0)
        trackball.setForwardScale(distance * 0.006)

        # Create a light if the model does not have one
        if not self.model_root.find('**/+Light'):
            self.light = self.render.attach_new_node(p3d.PointLight('light'))
            self.light.set_pos(0, -distance, distance)
            self.render.set_light(self.light)

        # Move lights to render
        self.model_root.clear_light()
        for light in self.model_root.find_all_matches('**/+Light'):
            light.parent.wrt_reparent_to(self.render)
            self.render.set_light(light)

        # Add some ambient light
        self.ambient = self.render.attach_new_node(p3d.AmbientLight('ambient'))
        self.ambient.node().set_color((.2, .2, .2,.2))
        self.render.set_light(self.ambient)

        if self.model_root.find('**/+Character'):
            self.anims = p3d.AnimControlCollection()
            p3d.autoBind(self.model_root.node(), self.anims, ~0)
            if self.anims.get_num_anims() > 0:
                self.anims.get_anim(0).loop(True)
        """
    def take_screenshot(self):
        
        Saves a screenshot of the current view to a timestamped PNG file.
        Uses Panda3D's built-in screenshot function.
        
        timestamp = p3d.Timestamp.now().get_rfc1123() # Get a timestamp for filename
        # defaultFilename=True generates names like screenshot_YYYY-MM-DD_HH-MM-SS.png
        # filename= can be used for a specific name, e.g., f"mesh_view_{timestamp}.png"
        self.screenshot(defaultFilename=True)
        print(f"Screenshot saved.") # Confirmation message
        """
    def toggle_normal_maps(self):
        self.pipeline.use_normal_maps = not self.pipeline.use_normal_maps

    def toggle_emission_maps(self):
        self.pipeline.use_emission_maps = not self.pipeline.use_emission_maps

    def toggle_occlusion_maps(self):
        self.pipeline.use_occlusion_maps = not self.pipeline.use_occlusion_maps

    def toggle_ambient_light(self):
        if self.render.has_light(self.ambient):
            self.render.clear_light(self.ambient)
        else:
            self.render.set_light(self.ambient)

def main():
    App().run()

if __name__ == '__main__':
    main()
