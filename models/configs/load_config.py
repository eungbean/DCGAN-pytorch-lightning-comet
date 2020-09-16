import argparse
import os.path
from .default import get_default_config


parser = argparse.ArgumentParser()

def load_config():
    cfg = get_default_config()
    
    print('successfully loaded config')
    # print(cfg)
    print('')

    return cfg