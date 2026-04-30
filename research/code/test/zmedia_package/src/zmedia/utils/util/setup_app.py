from pathlib import Path


def create_path_hirarchy(pth):
    fpth = Path(pth)
    try:
        fpth.mkdir(parents=True, exist_ok=True)
        print(f" created: {fpth}")
    except OSError as e:
        print(f"error creating: {fpth} with exception: {e}")    

def folder_setup( ap, dp, mp):
    
    # create app folders
    for p in ap:
      create_path_hirarchy(p)

    #create data folders

    for p in dp:
        create_path_hirarchy(p)

    #create model folders
    for p in mp:
        create_path_hirarchy(p)






