"""The server's 32B proves whenever the box is reachable; the local 8B only when it is not."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, prover   # noqa: E402

SERVER = ("http://127.0.0.1:8001/v1", "Goedel-LM/Goedel-Prover-V2-32B")


class Route(unittest.TestCase):
    def setUp(self):
        prover._SERVER_DOWN_UNTIL[0] = 0.0
        p = mock.patch.multiple(config, PROVER_FALLBACK_BASE=SERVER[0],
                                PROVER_FALLBACK_MODEL=SERVER[1])
        p.start()
        self.addCleanup(p.stop)
        e = mock.patch.dict(os.environ, {"GOEDEL_HOST": "gpu-box"})
        e.start()
        self.addCleanup(e.stop)

    def test_serving_server_wins(self):
        with mock.patch.object(prover, "_answers", return_value=True), \
             mock.patch.object(prover.subprocess, "run") as run:
            self.assertEqual(prover._prover_target(), SERVER)
            run.assert_not_called()

    def test_reachable_box_is_brought_up_not_skipped(self):
        answers = iter([False, True])        # down, then up after goedel_tunnel.sh
        ok = mock.Mock(returncode=0)
        with mock.patch.object(prover, "_answers", side_effect=lambda b: next(answers)), \
             mock.patch.object(prover.subprocess, "run", return_value=ok) as run:
            self.assertEqual(prover._prover_target(), SERVER)
            self.assertIn("up", run.call_args_list[-1].args[0])

    def test_dead_box_falls_back_once_then_backs_off(self):
        with mock.patch.object(prover, "_answers", return_value=False), \
             mock.patch.object(prover.subprocess, "run",
                               return_value=mock.Mock(returncode=255)) as run:
            local = (config.OLLAMA_BASE, config.PROVER_MODEL)
            self.assertEqual(prover._prover_target(), local)
            self.assertEqual(prover._prover_target(), local)
            self.assertEqual(run.call_count, 1, "a dead box was asked again inside the backoff")

    def test_no_host_never_runs_ssh(self):
        with mock.patch.dict(os.environ, {"GOEDEL_HOST": ""}), \
             mock.patch.object(prover, "_answers", return_value=False), \
             mock.patch.object(prover.subprocess, "run") as run:
            self.assertEqual(prover._prover_target(), (config.OLLAMA_BASE, config.PROVER_MODEL))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
