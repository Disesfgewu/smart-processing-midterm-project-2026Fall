import unittest
from check_change_size import violations


class ChangeSizeTests(unittest.TestCase):
    def check(self, pr_lines, commit_lines):
        return violations(
            {'additions': pr_lines, 'deletions': 0},
            [{'sha': 'abc', 'stats': {'additions': 0, 'deletions': commit_lines}}],
        )

    def test_boundary_allowed(self):
        self.assertEqual(self.check(300, 300), [])

    def test_pr_total_rejected(self):
        self.assertEqual(len(self.check(301, 100)), 1)

    def test_large_commit_not_hidden_by_reverts(self):
        self.assertEqual(len(self.check(1, 301)), 1)

    def test_both_limits_rejected(self):
        self.assertEqual(len(self.check(301, 301)), 2)


if __name__ == '__main__':
    unittest.main()
