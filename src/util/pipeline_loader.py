"""Explicit backbone selection; SD2 remains the default."""


def load_depth_pipeline(checkpoint, backbone="sd2", **kwargs):
    if backbone == "sd2":
        from marigold.marigold_pipeline import MarigoldPipeline
        pipeline_class = MarigoldPipeline
    elif backbone == "sdxl":
        from marigold.sdxl_depth_pipeline import SDXLDepthPipeline
        pipeline_class = SDXLDepthPipeline
    else:
        raise ValueError(f"Unknown depth backbone: {backbone}")
    return pipeline_class.from_pretrained(checkpoint, **kwargs)
