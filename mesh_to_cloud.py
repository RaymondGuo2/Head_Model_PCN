import sys

import open3d
import numpy as np
import os


def mesh_to_cloud(directory_path):
    for file in os.listdir(directory_path):
        if file.endswith("_processed.obj"):
            path = os.path.join(directory_path, file)
            mesh = open3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            # mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
            print(f"Number of Vertices for {file} = {len(np.asarray(mesh.vertices))}")
            print(f"Number of Triangles(faces) for {file} = {len(np.asarray(mesh.triangles))}")

            points = mesh.vertices
            point_cloud = open3d.geometry.PointCloud(points)

    # open3d.visualization.draw_geometries([point_cloud], width=1200, height=800)
            filename = os.path.splitext(file)[0] + '.pcd'
            cloud_file = os.path.join(directory_path, filename)
            open3d.io.write_point_cloud(cloud_file, point_cloud, write_ascii=True)


def visualise_pcd(pcd_file_path):
    pcd = open3d.io.read_point_cloud(pcd_file_path)
    print(np.asarray(pcd.points))
    open3d.visualization.draw_geometries([pcd], width=1200, height=800)


if __name__ == "__main__":
    visualise_pcd('./data/000031_processed.pcd')
