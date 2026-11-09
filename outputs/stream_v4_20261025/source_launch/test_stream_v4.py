import unittest

from turn_v2.stream_v4_data import causal_cases


class CausalDataTests(unittest.TestCase):
    def test_no_future_from_full_read_speech_message(self):
        row={"id":"fixture","language":"en","silence_spans":[{"start":.8,"end":1.2},{"start":2.,"end":2.5}],
             "words":[{"word":"I","end":.3},{"word":"want","end":.6},{"word":"coffee","end":1.8}],
             "messages":[{"role":"user","content":"I want coffee"}]}
        result=list(causal_cases(row,"fixture_read",False))
        self.assertEqual(result[0]["text"],"I want")
        self.assertEqual(result[0]["context"],"")
        self.assertEqual(result[0]["label"],"NEOT")
        self.assertEqual(result[1]["label"],"EOT")
        self.assertTrue(result[0]["current_turn_full_message_excluded"])

    def test_chinese_and_causal_cutoff(self):
        row={"id":"zh_fixture","language":"zh","silence_spans":[{"start":.5,"end":.8},{"start":1.6,"end":2.}],
             "words":[{"word":"我","end":.3},{"word":"要","end":.7},{"word":"咖啡","end":1.5}],"messages":[]}
        result=list(causal_cases(row,"fixture",False))
        self.assertEqual(result[0]["text"],"我要")
        self.assertEqual(result[1]["text"],"我要咖啡")
        self.assertEqual(result[0]["timestamp_seconds"],.7)


if __name__=="__main__":unittest.main()
