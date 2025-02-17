from typing import List
from pxr.Usd import Prim, Stage

RIGID_BODY_PRIMS = ('Mesh', 'Plane', 'Cube', 'Sphere', 'Cylinder', 'Capsule', 'Cone', 'Torus')


def get_prims_by_type(stage: Stage, prim_type: str) -> List[Prim]:
    """Get all prims of a certain type from a stage"""
    prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == prim_type:
            prims.append(prim)
    return prims


def get_prims_by_types(stage: Stage, prim_types: str) -> List[Prim]:
    """Get all prims that are one of certain types from a stage"""
    prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in prim_types:
            prims.append(prim)
    return prims
