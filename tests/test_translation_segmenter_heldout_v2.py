"""Test holdout overlap handling with synthetic text; no models or GPU."""
import json
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np

from turn_v2 import translation_segmenter_v15d_test_v2 as evaluation
from turn_v2.monolingual_data import encode_tail


class CharacterTokenizer:
    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2

    def encode(self, value, add_special_tokens=False):
        return [ord(char) + 3 for char in value]


class LeakageAuditTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("CREATE TABLE examples(language,split,record_json);"
                             "CREATE TABLE sources(language,page_id,text_sha256,split);")
        self.tokenizer = CharacterTokenizer()

    def tearDown(self):
        self.db.close()

    @staticmethod
    def row(context, source="vi:test", label="EOT"):
        return {"context": context, "text": "next", "source_id": source,
                "label": label, "lookahead_words": 1}

    def encode(self, rows):
        ids = np.full((len(rows), 256), 2, np.int32)
        lengths = np.zeros(len(rows), np.int64)
        for i, row in enumerate(rows):
            tokens = encode_tail(self.tokenizer, row["context"], row["text"])["input_ids"]
            ids[i, :len(tokens)] = tokens
            lengths[i] = len(tokens)
        return ids, lengths, np.array([r["label"] == "EOT" for r in rows]), np.ones(len(rows))

    def audit(self, rows):
        with patch.object(evaluation, "wait_for_gpu"), patch.object(evaluation, "status"):
            return evaluation.audit_test(self.db, "vi", rows, self.encode(rows), self.tokenizer)

    def test_detects_overlap_after_actual_context_truncation(self):
        train = self.row("training prefix " + "x" * 320, "vi:train")
        self.db.execute("INSERT INTO examples VALUES(?,?,?)", ("vi", "train", json.dumps(train)))
        rows = [self.row("different test prefix " + "x" * 320), self.row("unseen words", "vi:other")]
        clean, audit = self.audit(rows)
        self.assertEqual(clean.tolist(), [1])
        self.assertEqual(audit["overlapping_encoded_test_input_keys"], 1)

    def test_removes_whole_source_text_copy_even_if_inputs_differ(self):
        self.db.executemany("INSERT INTO sources VALUES(?,?,?,?)", [
            ("vi", "train", "same full text hash", "train"),
            ("vi", "copied", "same full text hash", "test"),
        ])
        clean, audit = self.audit([self.row("unseen excerpt", "vi:copied"), self.row("clean", "vi:clean")])
        self.assertEqual(clean.tolist(), [1])
        self.assertEqual(audit["test_source_groups_with_exact_non_test_text"], 1)

    def test_keeps_ambiguous_and_duplicate_labels_within_test(self):
        rows = [self.row("same test input", "vi:a", "EOT"), self.row("same test input", "vi:b", "NEOT")]
        clean, audit = self.audit(rows)
        self.assertEqual(clean.tolist(), [0, 1])
        self.assertEqual(audit["unique_test_model_input_keys"], 1)
        self.assertEqual(audit["removed_rows"], 0)

    def test_source_bootstrap_is_exact_on_a_perfect_classifier(self):
        rows = [self.row("a", "vi:a"), self.row("b", "vi:b")]
        scores = evaluation.cluster_intervals(rows, np.array([0, 1]), np.array([1, 0]), np.array([1, 0]))
        self.assertEqual(scores["source_groups"], 2)
        self.assertEqual(scores["accuracy"], [1.0, 1.0])
        # Some bootstrap draws contain only the negative source: F1 is then 0.
        self.assertEqual(scores["eot_f1"], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
