import bpy
import os
import re
from pathlib import Path

def export_as_gltf():
    filepath = bpy.data.filepath

    export_path = os.path.join(
        os.path.dirname(filepath),
        '../public/',
        Path(filepath).stem
    )
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        use_renderable=True,
        export_apply=True,
        export_hierarchy_full_collections=True,
        export_extras=True,
        export_cameras=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
    )

    print(f'Exported {filepath} to {export_path}')


export_as_gltf()
