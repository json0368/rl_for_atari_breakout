import numpy as np
from breakout_rl.envs import EnvConfig, make_env, frame_count
from breakout_rl.vendor.cleanrl.atari_wrappers import EpisodicLifeEnv


def test_real_ale_preprocessing_and_seed():
    config = EnvConfig(max_episode_frames=200)
    env, other = make_env(config, training=True), make_env(config, training=True)
    try:
        obs, _ = env.reset(seed=42)
        obs2, _ = other.reset(seed=42)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
        np.testing.assert_array_equal(obs, obs2)
        before = frame_count(env)
        for _ in range(100):
            new, reward, terminated, truncated, info = env.step(1)
            assert reward in (-1., 0., 1.)
            np.testing.assert_array_equal(new[:-1], obs[1:])
            obs = new
            if truncated:
                assert 'episode' in info
                break
            if terminated:
                # EpisodicLife may end a learning segment before the time limit.
                assert 'episode' not in info
                obs, _ = env.reset()
        else:
            raise AssertionError('time limit not reached')
        assert frame_count(env) > before
    finally:
        env.close(); other.close()


def test_life_loss_does_not_reset_game_or_raw_score_monitor():
    env = make_env(EnvConfig(max_episode_frames=10000), training=True)
    try:
        obs, _ = env.reset(seed=1)
        saw_life, saw_game = False, False
        for _ in range(4000):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                if 'episode' in info:
                    saw_game = True
                    assert info['episode']['l'] > 0
                    break
                saw_life = True
                lives = env.unwrapped.ale.lives()
                before = env.unwrapped.ale.getFrameNumber()
                obs, _ = env.reset()
                assert env.unwrapped.ale.lives() <= lives
                assert env.unwrapped.ale.getFrameNumber() > before
                np.testing.assert_array_equal(obs, np.repeat(obs[-1: ],4,axis=0))
        assert saw_life and saw_game
    finally:
        env.close()


def test_wrappers_match_pinned_upstream_pixels_and_endings():
    import gymnasium as gym
    from breakout_rl.vendor.cleanrl.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ClipRewardEnv
    reference = gym.make('BreakoutNoFrameskip-v4', frameskip=1, repeat_action_probability=0.0, max_episode_steps=5000)
    reference = gym.wrappers.RecordEpisodeStatistics(reference)
    reference = NoopResetEnv(reference, noop_max=30)
    reference = MaxAndSkipEnv(reference, skip=4)
    reference = EpisodicLifeEnv(reference)
    reference = FireResetEnv(reference)
    reference = ClipRewardEnv(reference)
    reference = gym.wrappers.ResizeObservation(reference,(84,84))
    reference = gym.wrappers.GrayscaleObservation(reference)
    reference = gym.wrappers.FrameStackObservation(reference,4)
    ours = make_env(EnvConfig(max_episode_frames=5000), training=True)
    try:
        a, _ = ours.reset(seed=7); b, _ = reference.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        rng=np.random.default_rng(91)
        for _ in range(500):
            action=int(rng.integers(4))
            a, ar, at, ax, ai=ours.step(action)
            b, br, bt, bx, bi=reference.step(action)
            np.testing.assert_array_equal(a,b)
            assert (ar,at,ax)==(br,bt,bx)
            assert ('episode' in ai)==('episode' in bi)
            if at or ax:
                a,_=ours.reset(); b,_=reference.reset()
                np.testing.assert_array_equal(a,b)
    finally:
        ours.close(); reference.close()
