import argparse
import pathlib as pth


def _species(laz):
    return laz.species


def _plotting_dependencies():
    import laspy
    import numpy as np

    if __package__:
        from .plot_cloud import plot_cloud
    else:
        from plot_cloud import plot_cloud

    return laspy, np, plot_cloud


def single_file(file_path: str | pth.Path):
    laspy, np, plot_cloud = _plotting_dependencies()
    laz = laspy.read(file_path)
    points = np.vstack((laz.x, laz.y, laz.z)).T

    print("Plotting semantic classification")
    plot_cloud(points, laz.classification)

    print("Plotting tree ids")
    plot_cloud(points, laz.tree_ids)

    print("Plotting species")
    plot_cloud(points, _species(laz))


def multiple_files(file_dir: str | pth.Path):
    laspy, np, plot_cloud = _plotting_dependencies()
    for file_path in pth.Path(file_dir).rglob("*.laz"):
        if "_mod" not in file_path.stem:
            continue

        laz = laspy.read(file_path)
        points = np.vstack((laz.x, laz.y, laz.z)).T
        points -= points.mean(axis=0)

        print(f"Plotting {file_path.stem}, semantic classification")
        plot_cloud(points, laz.classification)

        print(f"Plotting {file_path.stem}, tree ids")
        plot_cloud(points, laz.tree_ids)

        print(f"Plotting {file_path.stem}, species")
        plot_cloud(points, _species(laz))


def mutliple_files(file_dir: str | pth.Path):
    return multiple_files(file_dir)


def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Plot labels from LAZ files.")
    parser.add_argument(
        "--input-path",
        type=pth.Path,
        required=True,
        help="LAZ file or directory containing processed LAZ files.",
    )
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = argparser(argv)
    if args.input_path.is_file():
        single_file(args.input_path)
    elif args.input_path.is_dir():
        multiple_files(args.input_path)
    else:
        parser.error(f"Input path does not exist: {args.input_path}")


if __name__ == "__main__":
    main()
