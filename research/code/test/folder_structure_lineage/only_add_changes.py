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

'''

'''
def replicate_folders_files(root_src_dir,root_dst_dir):

    directories_added = []
    files_added = []
    memory_used = []

    memory_used.append(shutil.disk_usage(root_dst_dir))  
    for src_dir, dirs, files in os.walk(root_src_dir):
        print(src_dir, dirs, files)
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            directories_added.append(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                #if os.path.samefile(src_file, dst_file):
                    continue
                #os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
            files_added.append(dst_file)

    memory_used.append(shutil.disk_usage(root_dst_dir))        

    return (directories_added, files_added, memory_used)

def dir_copy(src, dst):
    try:
       shutil.copytree(src,dst)
       print(dst.split("\\")[-1]+ " ("+ dst.split("\\")[-2]+ ")"" copied!")
    
    except FileExistsError:
       pass


if __name__=='__main__':
    # print(copy_path())

    (d,f, m) = replicate_folders_files("/home/madhekar/temp/src", "/home/madhekar/temp/dest")
    print(f'directories added: {d} \n files added: {f} \n memory before: {m[0]} \n memory after: {m[1]}')


    #dir_copy("/home/madhekar/temp/src", "/home/madhekar/temp/dest")