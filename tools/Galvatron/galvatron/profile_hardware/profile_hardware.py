from galvatron.core import GalvatronProfiler, initialize_galvatron
import os

if __name__ == '__main__':
    args = initialize_galvatron(mode='profile_hardware')
    print(args)
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_path(path)
    
    # profile allreduce & p2p bandwidth
    profiler.profile_bandwidth()
    
    # profile overlapping slowdown coefficient
    profiler.profile_overlap()