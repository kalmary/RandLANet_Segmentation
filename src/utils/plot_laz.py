from plot_cloud import plot_cloud
import laspy
import numpy as np
import pathlib as pth


def _tree_species(laz):
    if "tree_species" in laz.point_format.extra_dimension_names:
        return laz.tree_species
    return laz.species


def single_file():
    laz = laspy.read("/Users/michalsiniarski/Documents/DATA/BRIK/GRAJEWO-TEST/ITWL_Grajewo20_1_mod.laz")
    points = np.vstack((laz.x, laz.y, laz.z)).T
    feature = laz.classification
    
    print("Plotting semantic classification")
    plot_cloud(points, feature)

    feature = laz.tree_ids
    print("Plotting tree ids")
    plot_cloud(points, feature)

    feature = _tree_species(laz)
    print("Plotting species")
    plot_cloud(points, feature)

def mutliple_files():
    
    file_dir = pth.Path("/mnt/SSD_EXT4_1TB/DATA/GRAJEWO_MINI")

    for file in file_dir.rglob("*.laz"):
        if "_mod" not in file.stem:
            continue
            
        laz = laspy.read(file)
        points = np.vstack((laz.x, laz.y, laz.z)).T
        points -= points.mean(axis=0)
        
        feature = laz.classification
        
        print(f"Plotting {file.stem}, semantic classification")
        plot_cloud(points, feature)


        feature = laz.tree_ids
        print(f"Plotting {file.stem}, tree ids")
        plot_cloud(points, feature)

        feature = _tree_species(laz)
        print(f"Plotting {file.stem}, species")
        plot_cloud(points, feature)


if __name__ == "__main__":
    single_file()
