from pathlib import Path
from direct.showbase.ShowBase import WindowControls
from panda3d.core import (
    Point3,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomVertexData,
    GeomNode,
    Geom,
    GeomTriangles,
    Vec3,
    NodePath,
    CullFaceAttrib,
)
import numpy as np


def create_box(dim, name="Box"):
    """
    Procedurally creates a textured 3D box geometry. Fixed.
    Panda3D Coords: +X=Right, +Y=Into Screen, +Z=Up
    width=X, height=Y, depth=Z
    """
    w, h, d = dim / 2.0
    vertices = [
        Point3(-w, -h, -d),
        Point3(w, -h, -d),
        Point3(w, h, -d),
        Point3(-w, h, -d),  # Bottom (-z)
        Point3(-w, -h, d),
        Point3(w, -h, d),
        Point3(w, h, d),
        Point3(-w, h, d),  # Top (+z)
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData(name + "Data", vformat, Geom.UH_static)
    vdata.set_num_rows(24)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    texcoord = GeomVertexWriter(vdata, "texcoord")
    tris = GeomTriangles(Geom.UH_static)

    def add_face(v_indices, norm_vec):
        start = vertex.get_write_row()
        vertex.add_data3(vertices[v_indices[0]])
        vertex.add_data3(vertices[v_indices[1]])
        vertex.add_data3(vertices[v_indices[2]])
        vertex.add_data3(vertices[v_indices[3]])
        for _ in range(4):
            normal.add_data3(norm_vec)
            texcoord.add_data2(uvs[_])
        tris.add_vertices(start + 0, start + 1, start + 2)
        tris.add_vertices(start + 2, start + 3, start + 0)
        tris.close_primitive()

    add_face([1, 0, 4, 5], Vec3(0, 0, -1))
    add_face([3, 2, 6, 7], Vec3(0, 0, 1))
    add_face([0, 3, 7, 4], Vec3(-1, 0, 0))
    add_face([2, 1, 5, 6], Vec3(1, 0, 0))
    add_face([0, 1, 2, 3], Vec3(0, -1, 0))
    add_face([5, 4, 7, 6], Vec3(0, 1, 0))
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(name + "GeomNode")
    node.add_geom(geom)
    nodepath = NodePath(node)
    nodepath.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
    return nodepath


def generate_random_box(seed, index, textures, render):
    """Creates a single 3D box, fixed at origin, with random texture/rotation."""
    # Random size for the main box
    rng = np.random.default_rng(seed)
    random_dim3 = rng.uniform(1, 4, 3)

    # Create main box geometry
    main_box_np = create_box(random_dim3, name=f"MainBox_{index}")

    # --- Position the box at the ORIGIN ---
    main_box_np.setPos(0, 0, 0)

    # Random rotation IS STILL APPLIED
    main_box_np.setHpr(rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360))

    # Apply random texture
    if textures:
        tex = rng.choice(textures)
        main_box_np.setTexture(tex)
    else:
        main_box_np.setColor(rng.random(), rng.random(), rng.random(), 1)

    # --- NO Patch is created on the 3D surface anymore ---

    # Attach the main box to the scene graph
    main_box_np.reparentTo(render)
    return main_box_np


def select_decal_texture(dir: Path, seed=10):
    rng = np.random.default_rng(seed)
    p = dir.glob("**/*")
    return rng.choice([x for x in p if x.is_file()], 1)[0]
