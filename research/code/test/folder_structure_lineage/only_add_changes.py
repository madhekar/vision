import os
import shutil

def copy_path(*, src='/home/madhekar/work/home-media-app/data/input-data/img', dst='/home/madhekar/temp/dest', dir_mode=0o777, follow_symlinks: bool = True):
    """
    Copy a source filesystem path to a destination path, creating parent
    directories if they don't exist.

    Args:
        src: The source filesystem path to copy. This must exist on the
            filesystem.

        dst: The destination to copy to. If the parent directories for this
            path do not exist, we will create them.

        dir_mode: The Unix permissions to set for any newly created
            directories.

        follow_symlinks: Whether to follow symlinks during the copy.

    Returns:
        Returns the destination path.
    """
    try:
        return shutil.copy2(src=src, dst=dst, follow_symlinks=follow_symlinks)
    except FileNotFoundError as exc:
        if exc.filename == dst and exc.filename2 is None:
            parent = os.path.dirname(dst)
            os.makedirs(name=parent, mode=dir_mode, exist_ok=True)
            return shutil.copy2(
                src=src,
                dst=dst,
                follow_symlinks=follow_symlinks,
            )
        raise


    

if __name__=='__main__':
  print(copy_path())