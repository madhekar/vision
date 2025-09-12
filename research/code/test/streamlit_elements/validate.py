import os
import time
import streamlit as st
from stqdm import stqdm
import multiprocessing as mp


st.set_page_config(
    page_title="zesha: Media Portal (MP)",
    #page_icon="../assets/zesha-high-resolution-logo.jpeg",  # check
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "About": "Zesha PC is created by Bhalchandra Madhekar",
        "Get Help": "https://www.linkedin.com/in/bmadhekar",
    },
)

def worker_function(item):
    """A function to be executed in a separate process."""
    time.sleep(0.1)  # Simulate some work
    return item * 2

@st.fragment
def test_validate_sqdm(npar):
    items = list(range(100))
    with mp.Pool(npar) as pool:
        results = list(stqdm(pool.imap(worker_function, items), total=len(items), desc="processing images"))
    st.write("validation check complete...")
    st.write(f"results: {results[:5]}...")  # Display first 10 results
    pool.close()
    pool.join()
    return results

@st.fragment
def test_duplicate_sqdm(npar):
    items = list(range(100))
    with mp.Pool(npar) as pool:
        results = list(stqdm(pool.imap(worker_function, items), total=len(items), desc="processing images"))
    st.write("duplicate check complete!")
    st.write(f"results: {results[:5]}...")  # Display first 10 results
    pool.close()
    pool.join()
    return results

@st.fragment
def test_quality_sqdm(npar):
    items = list(range(100))
    with mp.Pool(npar) as pool:
        results = list(stqdm(pool.imap(worker_function, items),total=len(items), desc="processing items"))
    st.write("quality check complete!")
    st.write(f"results: {results[:5]}...")  # Display first 10 results
    pool.close()
    pool.join()
    return results

@st.fragment
def test_metadata_sqdm(npar):
    items = list(range(100))
    with mp.Pool(npar) as pool:
            results = list(stqdm(pool.imap(worker_function, items),total=len(items),desc="processing items"))
    st.write("metadata check complete!")
    st.write(f"results: {results[:10]}...")  # Display first 10 results
    pool.close()
    pool.join()
    return results  

def execute(npar):

    #(dfi, dfv, dfd, dfa, dfn) = ss.extract_all_folder_stats(os.path.join(raw_data_path, user_source_selected))
    ca, cb, cc, cd = st.columns([1, 1, 1, 1], gap="small")
    with ca:
        ca.container(border=False)

    with cb:
        cb.container(border=False)

    with cc:
        cc.container(border=False)

    with cd:
        cd.container(border=False)            

    st.divider()    
    
    ###

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
    with c1:
        c1c = c1.container(border=False)
        with c1c:
            if st.button("Validation Check", use_container_width=True, type='primary'):
                with st.status('validate', expanded=True) as sc1c:
                    st.write('validation check start')
                    results = test_validate_sqdm(npar)
                    if results:
                        sc1c.update(label='validation complete...', state='complete')
                    else:
                        sc1c.update(label='validation failed...', state='error')   

    with c2:
        c2c= c2.container(border=False)
        with c2c:
            if  st.button("Duplicate Check", use_container_width=True, type='primary'):
                with st.status('duplicate', expanded=True) as sc2c:
                    st.write('duplicate image check start')
                    results = test_duplicate_sqdm(npar)
                    if results:
                        sc2c.update(label='duplicate complete...', state='complete')
                    else:
                        sc2c.update(label='duplicate failed...', state='error')  

    with c3:
        c3c= c3.container(border=False)
        with c3c:            
            if st.button("Quality Check", use_container_width=True, type='primary'):
              with st.status(label='quality', expanded=True) as sc3c:
                st.write('quality check start')
                results = test_quality_sqdm(npar)
                if results:
                    sc3c.update(label='quality complete', state='complete')
                else:
                    sc3c.update(label='quality failed', state='error')      

    with c4:
        c4c = c4.container(border=False)
        with c4c:
            if st.button("Metadata Check", use_container_width=True, type='primary'):
                with st.status(label='quality', expanded=True) as sc4c:
                   st.write('metadata check start')
                   results = test_metadata_sqdm(npar)
                   if results:
                       sc4c.update(label='metadata complete', state='complete')
                   else:
                       sc4c.update(label='metadata failed', state='error')  


if __name__=='__main__':
    cn = mp.cpu_count()
    execute(cn // 2)