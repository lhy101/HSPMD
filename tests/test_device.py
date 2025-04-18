import hspmd
import pickle
import unittest

class TestDevice(unittest.TestCase):

    _test_args = [
        ("cpu",), 
        ("cuda",), 
        ("cuda:1",), 
        ("localhost/cuda:1", 1), 
        ("server1/cuda:1",), 
        ("server2/cuda:1",)
    ]

    def test_device_getter(self):
        self.assertEqual(hspmd.device("cpu").type, "cpu")
        self.assertEqual(hspmd.device("cuda").type, "cuda")
        self.assertEqual(hspmd.device("cuda:2").type, "cuda")
        self.assertTrue(hspmd.device("cpu").is_cpu)
        self.assertFalse(hspmd.device("cpu").is_cuda)
        self.assertFalse(hspmd.device("cuda").is_cpu)
        self.assertTrue(hspmd.device("cuda").is_cuda)
        self.assertFalse(hspmd.device("cuda:1").is_cpu)
        self.assertTrue(hspmd.device("cuda:1").is_cuda)
        self.assertEqual(hspmd.device("cuda").index, 0)
        self.assertEqual(hspmd.device("cuda:0").index, 0)
        self.assertEqual(hspmd.device("cuda:1").index, 1)
        self.assertTrue(hspmd.device("cuda").local)
        self.assertTrue(hspmd.device("localhost/cuda").local)
        self.assertFalse(hspmd.device("server/cuda").local)
        self.assertEqual(hspmd.device("cuda:0").multiplex, 0)
        self.assertEqual(hspmd.device("cuda:0", multiplex=1).multiplex, 1)

    def test_device_cmp(self):
        devices = [hspmd.device(*args) for args in TestDevice._test_args]
        for i in range(len(TestDevice._test_args) - 1):
            self.assertEqual(devices[i], hspmd.device(*TestDevice._test_args[i]))
            self.assertLess(devices[i], devices[i + 1])

    def test_device_pickle(self):
        for args in TestDevice._test_args:
            device = hspmd.device(*args)
            device_dumped = pickle.dumps(device)
            device_loaded = pickle.loads(device_dumped)
            self.assertEqual(device, device_loaded)

    def test_device_group_pickle(self):
        device_group = hspmd.DeviceGroup(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        device_group_dumped = pickle.dumps(device_group)
        device_group_loaded = pickle.loads(device_group_dumped)
        self.assertEqual(device_group, device_group_loaded)

if __name__ == "__main__":
    unittest.main()

