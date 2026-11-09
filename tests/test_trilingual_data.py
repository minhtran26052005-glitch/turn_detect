import unittest

from turn_v2.trilingual_data import turn_rows, units


class TrilingualDataTests(unittest.TestCase):
    def test_crosswoz_usr_role_and_character_prefixes(self):
        turns = [("usr", "你好，我想找一家附近的川菜餐馆。"), ("sys", "请问您在哪个区？")]
        rows = list(turn_rows("crosswoz", "fixture", turns, "train"))
        self.assertEqual([r["label"] for r in rows], ["EOT", "NEOT", "NEOT"])
        self.assertEqual(rows[0]["language"], "zh")
        self.assertEqual(rows[0]["text"], turns[0][1])
        for row in rows[1:]:
            self.assertTrue(turns[0][1].startswith(row["text"]))
            self.assertLess(len(row["text"]), len(turns[0][1]))
            self.assertIn("semantically complete", row["label_limit"])

    def test_observed_context_contains_no_future_turn_or_hidden_suffix(self):
        turns = [("SYSTEM", "Where are you going?"), ("USER", "I would like a taxi to the airport please."), ("SYSTEM", "What time?")]
        rows = list(turn_rows("multiwoz", "fixture", turns, "dev"))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["context"] == "Where are you going?" for r in rows))
        self.assertTrue(all("What time?" not in r["text"] + r["context"] for r in rows))
        self.assertTrue(all(r["source_id"] == "multiwoz:fixture" for r in rows))
        self.assertEqual(rows[1]["label_source"], "observed_continuation_prefix_proxy")

    def test_cjk_is_not_one_whitespace_token(self):
        self.assertGreater(len(units("我想订一张北京到上海的火车票。", "zh")), 10)
        self.assertEqual(len(units("I need a ticket", "en")), 4)


if __name__ == "__main__": unittest.main()
