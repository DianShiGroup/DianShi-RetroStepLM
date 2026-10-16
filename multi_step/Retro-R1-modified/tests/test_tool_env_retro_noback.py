import unittest

from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tool_env_retro_noback import ToolEnvRetroNoBack, step


class StaticTool(Tool):
    def __init__(self, name, result=None):
        super().__init__(name, "test tool")
        self.result = result or {"results": []}

    def execute(self, args):
        return self.result


class MoleculeTool(Tool):
    def __init__(self, reactions):
        super().__init__("single_step_retro", "test tool")
        self.reactions = reactions

    def execute(self, args):
        return {"results": self.reactions.get(args["molecule"], [])}


class ForceNoLoopEmptyCandidateTest(unittest.TestCase):
    def test_filtered_empty_candidates_terminate_cleanly(self):
        retro = StaticTool(
            "single_step_retro",
            {"results": [[["CCC", "unavailable"]]]},
        )
        select = StaticTool("select_reaction")
        env = ToolEnvRetroNoBack(
            tools=[retro, select], force_noloop=True, max_turns=10
        ).copy("CC")
        env.unsolved_dict = {"0": ["CCC"], "1": ["CC"]}
        env.step_dict = {"0": 0, "1": 1}
        env.current_state = 1

        observation, _reward, _done, info = step(
            env,
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"1-0"}}</tool_call>',
        )

        self.assertTrue(info["action_is_effective"])
        self.assertIn("No reactions.", observation)
        self.assertTrue(env.back_flag)
        self.assertEqual(env.reaction_dict["1-0"], [])

        error, valid = env._process_action_select("1-0-0")
        self.assertFalse(valid)
        self.assertIn("no candidate reactions", error)
        self.assertNotIn("--1", error)


class SiblingRetryTest(unittest.TestCase):
    def test_tries_all_siblings_after_a_dead_end(self):
        single_step = MoleculeTool(
            {
                "P": [
                    [["A", "unavailable"]],
                    [["B", "unavailable"]],
                    [["C", "available"]],
                ],
                "A": [],
                "B": [],
            }
        )
        select = StaticTool("select_reaction")
        env = ToolEnvRetroNoBack(
            tools=[single_step, select],
            max_turns=10,
            sibling_retry=True,
        ).copy("P")

        observation, _reward, _done, _info = step(
            env,
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"0-0"}}</tool_call>',
        )
        self.assertIn("Reaction 0-0-0", observation)

        observation, _reward, _done, info = step(
            env,
            '<tool_call>{"name":"select_reaction","arguments":{"reaction":"0-0-0"}}</tool_call>',
        )
        self.assertTrue(info["action_is_effective"])
        self.assertIn("Molecule 1-0: A", observation)

        observation, _reward, _done, info = step(
            env,
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"1-0"}}</tool_call>',
        )
        self.assertTrue(info["action_is_effective"])
        self.assertFalse(env.back_flag)
        self.assertEqual(env.count_sibling_retry, 1)
        self.assertIn("Automatic sibling retry selected reaction 0-0-1", observation)
        self.assertIn("Molecule 2-0: B", observation)

        observation, _reward, _done, info = step(
            env,
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"2-0"}}</tool_call>',
        )
        self.assertTrue(info["action_is_effective"])
        self.assertTrue(env.back_flag)
        self.assertEqual(env.count_sibling_retry, 2)
        self.assertIn("Automatic sibling retry selected reaction 0-0-2", observation)
        self.assertIn("No unsolved molecules", observation)

    def test_fails_only_after_every_sibling_is_exhausted(self):
        single_step = MoleculeTool(
            {
                "P": [
                    [["A", "unavailable"]],
                    [["B", "unavailable"]],
                    [["C", "unavailable"]],
                ],
                "A": [],
                "B": [],
                "C": [],
            }
        )
        env = ToolEnvRetroNoBack(
            tools=[single_step, StaticTool("select_reaction")],
            sibling_retry=True,
        ).copy("P")

        actions = [
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"0-0"}}</tool_call>',
            '<tool_call>{"name":"select_reaction","arguments":{"reaction":"0-0-0"}}</tool_call>',
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"1-0"}}</tool_call>',
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"2-0"}}</tool_call>',
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"3-0"}}</tool_call>',
        ]
        for action in actions:
            observation, _reward, _done, _info = step(env, action)

        self.assertTrue(env.back_flag)
        self.assertEqual(env.count_sibling_retry, 2)
        self.assertEqual(env._selection_stack, [])
        self.assertIn("No reactions", observation)
        self.assertIn("exhausted the available sibling reactions", observation)

    def test_config_switch_disables_retry(self):
        single_step = MoleculeTool(
            {
                "P": [
                    [["A", "unavailable"]],
                    [["B", "available"]],
                ],
                "A": [],
            }
        )
        env = ToolEnvRetroNoBack(
            tools=[single_step, StaticTool("select_reaction")],
            sibling_retry=False,
        ).copy("P")

        actions = [
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"0-0"}}</tool_call>',
            '<tool_call>{"name":"select_reaction","arguments":{"reaction":"0-0-0"}}</tool_call>',
            '<tool_call>{"name":"single_step_retro","arguments":{"molecule":"1-0"}}</tool_call>',
        ]
        for action in actions:
            observation, _reward, _done, _info = step(env, action)

        self.assertTrue(env.back_flag)
        self.assertEqual(env.count_sibling_retry, 0)
        self.assertNotIn("Automatic sibling retry", observation)


if __name__ == "__main__":
    unittest.main()
