import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def find_nearest_point(source_point, target_points):
    distances = np.linalg.norm(target_points - source_point, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index


def main():
    # Load the source and target .ply files
    source_file = PlyData.read(
        "/home/mbortolon/data/pose-splatting/toy_sample/point_cloud.ply"
    )
    target_file = PlyData.read(
        "/home/mbortolon/data/pose-splatting/toy_sample/point_cloud_reduce_further.ply"
    )

    # Extract point data from the source and target files
    source_points = np.vstack(
        (
            source_file["vertex"]["x"],
            source_file["vertex"]["y"],
            source_file["vertex"]["z"],
        )
    ).T
    target_points = np.vstack(
        (
            target_file["vertex"]["x"],
            target_file["vertex"]["y"],
            target_file["vertex"]["z"],
        )
    ).T

    # Initialize a list to store the resulting points
    result_points = []

    name_properties = [
        element_property.name for element_property in source_file.elements[0].properties
    ]

    # Iterate through each point in the source file and find the nearest point in the target file
    for target_point in tqdm(target_points):
        nearest_point_idx = find_nearest_point(target_point, source_points)
        result_points.append(
            tuple(
                [
                    source_file["vertex"][name_property][nearest_point_idx]
                    for name_property in name_properties
                ]
            )
        )

    # Create a new .ply file with the result points and attributes
    result_data = np.array(
        result_points,
        dtype=[
            (element_property.name, element_property.dtype())
            for element_property in source_file.elements[0].properties
        ],
    )
    print("Ciao")

    result_element = PlyElement.describe(result_data, source_file.elements[0].name)
    PlyData(
        [result_element],
        text=source_file.text,
        byte_order=source_file.byte_order,
        comments=source_file.comments,
        obj_info=source_file.obj_info,
    )
    result_file = PlyData([result_element])

    # Save the result to a new .ply file
    result_file.write(
        "/home/mbortolon/data/pose-splatting/toy_sample"
        "/point_cloud_reduce_further_complete.ply"
    )


if __name__ == "__main__":
    main()
