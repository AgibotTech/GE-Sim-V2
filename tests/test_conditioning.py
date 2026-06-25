import numpy as np


def test_render_band_from_bundle(demo_bundle_dir):
    from gesim.conditioning.band import render_band_from_bundle
    from gesim.episode import EpisodeBundle

    bundle = EpisodeBundle.load(demo_bundle_dir)
    band, c2w = render_band_from_bundle(bundle)
    band = np.asarray(band)
    assert band.ndim == 5 and band.shape[0] == 3
    assert float(band.min()) >= 0.0 and float(band.max()) <= 1.0
    v, t = band.shape[1], band.shape[2]
    assert c2w.shape == (v, t, 4, 4)


def test_policy_band_renderer(demo_bundle_dir):
    from gesim.conditioning.policy_band import PolicyBandRenderer
    from gesim.episode import EpisodeBundle

    bundle = EpisodeBundle.load(demo_bundle_dir)
    renderer = PolicyBandRenderer(bundle)  # bundled compiled Genie-01 (G01) kinematics
    assert bundle.actions is not None
    actions = bundle.actions[:25]
    band, c2w = renderer.render(actions)
    assert np.asarray(band).shape[2] == 25
    assert c2w.shape[1] == 25
