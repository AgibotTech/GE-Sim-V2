"""Use the existing video trunk without unused Pose Expert buffer parameters."""

from gesim.models.gesim_v2.networks.transformers.transformer_cosmos_multiview_PE import (
    MultiViewCosmosTransformer3DModelState,
)


class VideoOnlyTransformer(MultiViewCosmosTransformer3DModelState):
    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "pose_expert": False})
        # The shared class allocates these even with pose_expert=False. Teacher
        # checkpoints never contain them, and video generation never uses them.
        del self.video_buffer_weights
        del self.video_buffer_norm

    def forward(self, *args, **kwargs):
        if kwargs.get("return_pose", False):
            raise ValueError("This checkpoint has no Pose Expert")
        kwargs["store_buffer"] = False
        return super().forward(*args, **kwargs)
