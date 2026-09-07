import hashlib

import numpy as np
import sapien.core as sapien


engine = sapien.Engine()
from sapien.render import set_global_config

set_global_config(max_num_materials=50000, max_num_textures=50000)
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(4)
sapien.render.set_ray_tracing_path_depth(2)
sapien.render.set_ray_tracing_denoiser("oidn")

scene = engine.create_scene(sapien.SceneConfig())
scene.set_ambient_light([0.2, 0.2, 0.2])
scene.add_directional_light([1.0, 0.0, -1.0], [1.0, 1.0, 1.0], shadow=True)
scene.add_ground(0.0)
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.4, 0.4, 0.4], material=[0.9, 0.1, 0.1])
actor = builder.build_static(name="probe_cube")
actor.set_pose(sapien.Pose([0.0, 0.0, 0.4]))
camera = scene.add_camera("probe", 96, 64, 0.9, 0.1, 20.0)
camera.set_pose(sapien.Pose([-3.0, 0.0, 0.8]))

scene.update_render()
camera.take_picture()
color = np.asarray(camera.get_float_texture("Color"), dtype=np.float32)
if color.shape != (64, 96, 4):
    raise RuntimeError(f"unexpected render shape: {color.shape}")
if not np.isfinite(color).all():
    raise RuntimeError("render contains non-finite values")
rgb = color[..., :3]
if float(rgb.max() - rgb.min()) <= 1e-4:
    raise RuntimeError("render is blank")
print(
    {
        "shape": list(color.shape),
        "minimum": float(rgb.min()),
        "maximum": float(rgb.max()),
        "sha256": hashlib.sha256(color.tobytes(order="C")).hexdigest(),
    }
)
