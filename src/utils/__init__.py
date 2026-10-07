def __getattr__(name):
    if name == "plot_cloud":
        from .plot_cloud import plot_cloud

        return plot_cloud
    raise AttributeError(name)
