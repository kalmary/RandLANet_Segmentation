from plot_cloud import plot_cloud
import laspy
import numpy as np


def main():
    file_path = "/mnt/SSD_EXT4_1TB/DATA/GRAJEWO_MINI_TEST/ITWL_Grajewo20_mini_mod.laz"
    laz = laspy.read(file_path)
    points = np.vstack((laz.x, laz.y, laz.z)).T
    intensity = laz.species
    print(np.unique(intensity))

    plot_cloud(points, intensity)

if __name__ == "__main__":
    main()