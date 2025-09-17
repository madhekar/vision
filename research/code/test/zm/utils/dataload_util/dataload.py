import os
import shutil
import uuid
from tqdm import tqdm
from utils.config_util import config
from utils.util import file_type_ext as fte
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))

'''
restore clean target media folders
'''
def clean_media_folders(folder):
    # create clean destination folders
    sm.add_messages("validate", f"s|starting to clean {folder}\n\n")
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
    sm.add_messages("validate", f"s|done to cleaning {folder}\n\n")    


def handle_copy_media_files(root, fdest_media, uuid_path, media_items):

    f_dest = os.path.join(fdest_media, uuid_path)
    os.makedirs(f_dest)
    #sm.add_messages('load', f's| {root} -> {fdest_media} coping {len(media_items)}# files.\n')
    
    for item in media_items:
        item_path = os.path.join(root, item)

        if os.path.isfile(item_path):
            try:
                shutil.copy(item_path, f_dest)
            except FileNotFoundError:
                e1 = ReferenceError("Source file not found.")
                sm.add_messages("validate",f"e|exception: {e1} Source file not found {item_path}\n\n")
                continue
            except PermissionError:
                e2 = RuntimeError("Permission denied.")
                sm.add_messages("validate",f"e|exception: {e2} file permissing denied {item_path}")
                continue
            except FileExistsError:
                e3 = RuntimeError("Destination file already exists.")
                sm.add_messages("validate", f"e|exception: {e3} destination file: {f_dest} already exists.\n\n")
                continue
            except Exception as e:
                e4 = RuntimeError(f"An Unknown error occurred in datavalidate: {e}")
                sm.add_messages("validate", f"e|exception: {e4} unknown file exception: {f_dest}.\n\n")
                continue

## possible performance issue 
def copy_files_only(src_dir, fdest_image, fdest_txt, fdest_video, fdest_audio ):

    img_items, txt_items, vid_items, adu_items = ([] for i in range(4))

    #create clean destination folders
    clean_media_folders(fdest_image)
    clean_media_folders(fdest_txt)
    clean_media_folders(fdest_video)
    clean_media_folders(fdest_audio)
    
    #total files
    total_items = 0
    for dp, dns, fns in os.walk(src_dir):
        total_items += len(dns) + len(fns)

    with tqdm(total=total_items, desc=f'processing: {src_dir}') as pbar:
        for root, dirnames, items in os.walk(src_dir):
            if not dirnames:
                if len(items) > 0:
                    print(root + " - " + str(dirnames) + " - " + str(items))
                    img_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.image_types]
                    vid_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.video_types]
                    txt_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.document_types]
                    adu_items = [f for f in items if os.path.splitext(f)[1].lower() in fte.audio_types]

                    #generate uuid path
                    uuid_path = path_encode(root)
                    
                    # handle image items
                    if len(img_items) > 0:
                        handle_copy_media_files(root, fdest_image, uuid_path, img_items)

                    if len(vid_items) > 0:
                        handle_copy_media_files(root, fdest_video, uuid_path, vid_items)    

                    if len(txt_items) > 0:
                        handle_copy_media_files(root, fdest_txt, uuid_path, txt_items)

                    if len(adu_items) > 0:
                        handle_copy_media_files(root, fdest_audio, uuid_path, adu_items)     
                    
                    pbar.update(len(items))
'''
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img/'
  input_video_path: '/home/madhekar/work/home-media-app/data/input-data/video/'
  input_txt_path: '/home/madhekar/work/home-media-app/data/input-data/txt/'
  input_audio_path: '/home/madhekar/work/home-media-app/data/input-data/audio/'
'''
def clean_unknown_files_folders(fdest_image, fdest_txt, fdest_video, fdest_audio):
    
    ifcnt = ss.trim_unknown_files(fdest_image)        
    tfcnt = ss.trim_unknown_files(fdest_txt)
    vfcnt = ss.trim_unknown_files(fdest_video)
    afcnt = ss.trim_unknown_files(fdest_audio)
   
    idcnt = ss.remove_empty_folders(fdest_image)
    tdcnt = ss.remove_empty_folders(fdest_txt)
    vdcnt = ss.remove_empty_folders(fdest_video)
    adcnt = ss.remove_empty_folders(fdest_audio)

    sm.add_messages("validate", f"s| file:folder cleanup- image:{ifcnt}:{idcnt} text: {tfcnt}:{tdcnt} video: {vfcnt}:{vdcnt} audio: {afcnt}:{adcnt} \n\n")

def execute(source_name):
    (
        raw_data_path,
        input_image_path,
        input_txt_path,
        input_video_path,
        input_audio_path
    ) = config.dataload_config_load()

    """
      paths to import files
    """
    ipath = os.path.join(input_image_path, source_name)
    tpath = os.path.join(input_txt_path, source_name)
    vpath = os.path.join(input_video_path, source_name)
    apath = os.path.join(input_audio_path, source_name)

    copy_files_only(raw_data_path, ipath, tpath, vpath, apath)

    clean_unknown_files_folders(ipath, tpath, vpath, apath)
    # source_list = []
    # # source_list = get_external_devices(get_user())
    # source_list = mu.extract_user_raw_data_folders(raw_data_path)
    # if len(source_list) > 0:
    #     ext = st.sidebar.selectbox(label="Select Source", options=source_list)

    # st.sidebar.caption("CHECK FOLDERS TO TRIM", unsafe_allow_html=True)
    # display_folder_tree(get_path_as_dict(os.path.join(raw_data_path, ext)))
    # st.sidebar.button(label="TRIM CHECKED FOLDERS", use_container_width=True)
    # # c1.text_area(label="External Source Structure", value= display_tree(os.path.join('/media/madhekar/' , ext)))


if __name__ == "__main__":
    execute()