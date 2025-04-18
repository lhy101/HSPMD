import hspmd
import pickle
import unittest
import numpy as np

class TestStream(unittest.TestCase):

    _test_args = [
        ("cpu", 0), 
        ("cuda", 1), 
        ("cuda:1", 2), 
    ]

    def test_stream_getter(self):
        self.assertEqual(hspmd.stream("cuda:1", 0).device, hspmd.device("cuda:1"))
        self.assertEqual(hspmd.stream("cpu", 0).device_type, "cpu")
        self.assertEqual(hspmd.stream("cuda", 0).device_type, "cuda")
        self.assertEqual(hspmd.stream("cuda", 0).device_index, 0)
        self.assertEqual(hspmd.stream("cuda:2", 0).device_index, 2)
        self.assertEqual(hspmd.stream("cuda", 0).stream_index, 0)
        self.assertEqual(hspmd.stream("cuda", 3).stream_index, 3)

    def test_device_cmp(self):
        streams = [hspmd.stream(*args) for args in TestStream._test_args]
        for i in range(len(TestStream._test_args) - 1):
            self.assertEqual(streams[i], hspmd.stream(*TestStream._test_args[i]))
            with self.assertRaises(RuntimeError):
                self.assertLess(streams[i], streams[i + 1])

    def test_stream_pickle(self):
        for args in TestStream._test_args:
            stream = hspmd.stream(*args)
            stream_dumped = pickle.dumps(stream)
            stream_loaded = pickle.loads(stream_dumped)
            self.assertEqual(stream, stream_loaded)

if __name__ == "__main__":
    unittest.main()

