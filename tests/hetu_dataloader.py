import hspmd
import numpy as np

if __name__ == "__main__":
    a = np.random.rand(7, 2).astype(np.float32)
    b = np.random.rand(2, 4).astype(np.float32)
    a_nd = hspmd.NDArray(a, dtype=hspmd.float32, device="cpu")
    x = None
    with hspmd.graph("eager"):
        with hspmd.context(eager_device="cpu"):
            x = hspmd.from_numpy(b)
            mm = hspmd.utils.data.DataLoader(a_nd, batch_size = 2, shuffle = True)
    print(a_nd)
    import time
    print(len(mm))
    with hspmd.graph("eager"):
        with hspmd.context(eager_device="cpu"):
            for u in mm:
                print(u.shape)
                out = hspmd.matmul(u, x)
                print(out.numpy(force=True))
    
    print("Thank you")