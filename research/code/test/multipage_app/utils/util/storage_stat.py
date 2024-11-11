import glob

def get_file_cnt(spath):
  img_ctr = len(glob.glob1(spath, "*.jpg"))