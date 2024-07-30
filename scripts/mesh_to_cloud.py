import open3d
import numpy as np
import os
from util import unit_sphere_normalisation


# Function to convert obj mesh files to point cloud
def mesh_to_cloud(directory_path, preprocess=True, training_resolution=2048):
    directory_path = os.path.expanduser(directory_path)
    for file in os.listdir(directory_path):
        if file.endswith(".obj"):
            path = os.path.join(directory_path, file)
            mesh = open3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            points = mesh.vertices
            npy_array = np.asarray(points)

            print(f"Number of Vertices for {file} = {len(np.asarray(mesh.vertices))}")
            print(f"Number of Triangles(faces) for {file} = {len(np.asarray(mesh.triangles))}")
            # Preprocess points according to Stasinakis et al. (2021)
            if preprocess:
                if training_resolution == 2048:
                    poisson_pcd = mesh.sample_points_poisson_disk(10000)
                elif training_resolution == 16384:
                    poisson_pcd = mesh.sample_points_poisson_disk(30000)
                else:
                    raise ValueError(f"Unsupported training resolution {training_resolution}")
                poisson_fps_pcd = poisson_pcd.farthest_point_down_sample(training_resolution)
                points = poisson_fps_pcd.points
                # Convert point data to numpy array
                poisson_fps_pcd = np.asarray(points)
                npy_array = unit_sphere_normalisation(poisson_fps_pcd)

            point_cloud = open3d.geometry.PointCloud(points)
            cloud_filename = os.path.splitext(file)[0] + '.pcd'
            cloud_file = os.path.join(directory_path, cloud_filename)
            open3d.io.write_point_cloud(cloud_file, point_cloud, write_ascii=True)
            print(f"Saved point cloud {cloud_filename}")

            numpy_filename = os.path.splitext(file)[0] + '.npy'
            numpy_file = os.path.join(directory_path, numpy_filename)
            np.save(numpy_file, npy_array)
            print(f"Saved numpy file {numpy_filename}")


def visualise_pcd(pcd_file_path, training_resolution=16384):
    pcd = open3d.io.read_point_cloud(pcd_file_path)
    open3d.visualization.draw_geometries([pcd], width=1200, height=800)


if __name__ == "__main__":
    # Convert the preprocessed mesh files into point clouds
    mesh_to_cloud('~/../../vol/bitbucket/rqg23/preprocessed_obj_partial_inputs')

    # Visualise the point clouds
    # visualise_pcd('./data/000031_processed.pcd')

    # Test the shape of the numpy input
    # file = '../data/000007_processed.npy'
    # trial_file = np.load(file)
    # print(trial_file.shape)
