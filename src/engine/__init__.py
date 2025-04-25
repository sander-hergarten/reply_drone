import trimesh

def main():
    mesh = trimesh.load("odm_mesh.ply")
    print("Loaded ply")
    mesh.export("odm_mesh.gltf")
    print("Exported to gltf")

if __name__ == "__main__":
    main()
