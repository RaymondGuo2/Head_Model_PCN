import open3d
import numpy as np
import os
import trimesh


# Various parts taken from Open3D documentation: https://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html#from-numpy-to-open3d-pointcloud
def numpy_to_cloud(file_path):
    numpy_array = np.load(file_path)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(numpy_array)

    path = os.path.splitext(file_path)[0] + 'post.pcd'
    open3d.io.write_point_cloud(path, pcd)

    pcd_load = open3d.io.read_point_cloud(path)
    open3d.visualization.draw_geometries([pcd_load])


# Code derived from https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices
def cloud_to_mesh(file_path):
    pcd = open3d.io.read_point_cloud(file_path)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        open3d.utility.DoubleVector([radius, radius * 2])
    )

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    trimesh.convex.is_convex(tri_mesh)

    open3d.io.write_triangle_mesh('../data/processed.obj', mesh)
    open3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    # file_path = './data/000031_processed.npy'
    # numpy_to_cloud(file_path)
    cloud_to_mesh('../data/000007_processedpost.ply')

