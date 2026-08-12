from __future__ import annotations

import unittest

from safety_data.ppo_sqrl_index import INDEX_SCHEMA, nested_episode_selections


class PpoSqrlIndexTest(unittest.TestCase):
    def test_nested_selection_preserves_whole_episodes_and_strata(self):
        episodes = []
        for seed in (137, 138):
            for stage in ("early", "boundary", "mature"):
                for failed in (False, True):
                    for index in range(10):
                        episodes.append({
                            "key": f"{seed}:{stage}:{failed}:{index}",
                            "seed": seed,
                            "stage": stage,
                            "fall": failed,
                            "transitions": 10,
                        })
        value = nested_episode_selections({
            "schema_version": INDEX_SCHEMA,
            "episodes": episodes,
            "transition_count": 1200,
        }, budgets=(240, 600, 960))
        sets = [set(item["episode_keys"]) for item in value["selections"]]
        self.assertTrue(sets[0] < sets[1] < sets[2])
        self.assertEqual(
            [item["realized_transitions"] for item in value["selections"]],
            [240, 600, 960],
        )


if __name__ == "__main__":
    unittest.main()
