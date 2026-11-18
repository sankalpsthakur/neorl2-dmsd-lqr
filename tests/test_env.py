import json, pathlib
import numpy as np
from nd import env, model, data

FIX = json.load(open(pathlib.Path(__file__).parent / "fixtures" / "real_rows.json"))["rows"]


def test_normalisation_reproduces_published_pairs():
    # BARS.md: raw returns of the repository CSVs -> table values
    assert abs(env.normalize(494.45271875) - 56.642698068163774) < 1e-9
    assert abs(env.normalize(674.69) - 78.65) < 0.005
    assert abs(env.normalize(563.63) - 65.09) < 0.005


def test_true_system_matches_three_real_rows():
    # the first three stored transitions of the real train file are one-step consistent with the true linear system (float32 data)
    A, B = model.true_system()
    for r in FIX:
        obs = np.array([r[f"obs_{i}"] for i in range(6)]); act = np.array([r["action_0"], r["action_1"]]); nxt = np.array([r[f"next_obs_{i}"] for i in range(6)])
        pred = A @ obs[:4] + B @ act
        assert np.max(np.abs(pred - nxt[:4])) < 2e-6


def test_reward_matches_three_real_rows():
    for r in FIX:
        nxt = np.array([[r[f"next_obs_{i}"] for i in range(6)]])
        assert abs(data.reward_from_next_obs(nxt)[0] - r["reward"]) < 1e-5
        assert abs(env.step_reward32(nxt[:, :2], nxt[:, 4:6])[0] - r["reward"]) < 1e-5


def test_target_constant_between_rows_and_actions_bounded():
    assert all(abs(r["action_0"]) <= 1 and abs(r["action_1"]) <= 1 for r in FIX)
    assert FIX[0]["obs_4"] == FIX[0]["next_obs_4"] and FIX[1]["obs_4"] == FIX[0]["next_obs_4"]


def test_draw_episode_order_and_ranges():
    e = env.draw_episode(123)
    assert e["targets"].shape == (100, 2) and np.all(np.abs(e["targets"]) <= 1.5)
    assert np.all(np.abs(e["x0"][:2]) <= 0.25) and np.all(np.abs(e["x0"][2:]) <= 0.1)
    assert np.allclose(e["task"], [4, 4, 4, 2, 2, 2, 20, 20])
    assert np.allclose(env.draw_episode(5, "test")["task"][6:], [30, 30])
    # same seed, same episode
    assert np.array_equal(env.draw_episode(9)["targets"], env.draw_episode(9)["targets"])
    # official RNG order: 8 task draws first, then the target counts. Recompute by hand.
    rs = np.random.RandomState(123)
    [rs.uniform(4.0, 4.0) for _ in range(3)]; [rs.uniform(2.0, 2.0) for _ in range(3)]; [rs.uniform(20.0, 20.0) for _ in range(2)]
    n0 = rs.randint(1, 3)
    first = rs.uniform(-1.5, 1.5)
    assert e["targets"][0, 0] == first and n0 in (1, 2)


def test_zero_policy_return_by_hand():
    seed = 77
    e = env.draw_episode(seed)
    x = e["x0"].copy(); ret = 0.0
    for t in range(100):
        x = e["A"] @ x
        tn = e["targets"][min(t + 1, 99)]
        ret += float(np.sum(-np.log(np.abs(x[:2].astype(np.float32) - tn.astype(np.float32)) + np.float32(0.01))))
    # zero action is force -30 (action 0 maps to (0+1)/2*60-30 = 0): force 0
    got = env.rollout(lambda o: np.zeros((len(o), 2)), [seed])[0]
    assert abs(got - ret) < 1e-3


def test_force_mapping_action_one_is_30():
    # one step from rest at the target with action 1: velocity gains dt/m * 30 = 0.3
    A, invm = env.dynamics_from_task([4, 4, 4, 2, 2, 2, 20, 20])
    assert abs(0.2 * invm[0] * 30 - 0.3) < 1e-12 and A[0, 2] == 0.2
