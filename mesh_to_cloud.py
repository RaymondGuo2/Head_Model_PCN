import numpy as np
import open3d
from pyvirtualdisplay import Display

display = Display(visible=0, size=(800, 600))
display.start()

obj_file = "/homes/rqg23/individualproject/object_autocompletion/data/000007_processed.obj"
mesh = open3d.io.read_triangle_mesh(obj_file)
mesh.compute_vertex_normals()
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),center=mesh.get_center())
print(f"Number of Vertices = {len(np.asarray(mesh.vertices))}")
print(f"Number of Triangles(faces) = {len(np.asarray(mesh.triangles))}")

points = mesh.vertices
point_cloud = open3d.geometry.PointCloud(points)

open3d.visualization.draw_geometries([point_cloud], width=1200, height=800)
open3d.io.write_point_cloud("/homes/rqg23/individualproject/object_autocompletion/data/000007_processed.pcd", point_cloud, write_ascii=True)

display.stop()