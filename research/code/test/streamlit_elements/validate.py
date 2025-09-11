import streamlit as st
from stqdm import stqdm
import multiprocessing as mp
import time

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


def test_validate_sqdm(npar):
    items = list(range(500))
    with mp.Pool(npar) as pool:
        # Use stqdm to wrap the results from the multiprocessing pool
        st.write('validation started...')
        results = list(stqdm(pool.imap(worker_function, items), total=len(items), desc="Processing items"))
    st.write("Processing complete...")
    st.write(f"Results: {results[:10]}...")  # Display first 10 results
    return results


def test_duplicate_sqdm(npar):
    items = list(range(100))


    with mp.Pool(npar) as pool:
        # Use stqdm to wrap the results from the multiprocessing pool
        results = list(
            stqdm(
                pool.imap(worker_function, items),
                total=len(items),
                desc="Processing items",
            )
        )
    st.write("Processing complete!")
    st.write(f"Results: {results[:10]}...")  # Display first 10 results


def test_quality_sqdm(npar):
    items = list(range(100))

    with mp.Pool(npar) as pool:
        # Use stqdm to wrap the results from the multiprocessing pool
        results = list(
            stqdm(
                pool.imap(worker_function, items),
                total=len(items),
                desc="Processing items",
            )
        )
    st.write("Processing complete!")
    st.write(f"Results: {results[:10]}...")  # Display first 10 results

def test_metadata_sqdm(npar):
    #st.subheader("Multiprocessing with stqdm in Streamlit")

    items = list(range(100))

    if st.button("Metadata Processing"):
        with mp.Pool(npar) as pool:
            # Use stqdm to wrap the results from the multiprocessing pool
            results = list(
                stqdm(
                    pool.imap(worker_function, items),
                    total=len(items),
                    desc="Processing items",
                )
            )
        st.write("Processing complete!")
        st.write(f"Results: {results[:10]}...")  # Display first 10 results
      

def execute(npar):

    ca, cb, cc, cd = st.columns([1, 1, 1, 1], gap="small")
    with ca:
        ca.container(border=False)

    with cb:
        cb.container(border=True)

    with cc:
        cc.container(border=True)

    with cd:
        cd.container(border=True)            

    st.divider()    
    
    ###
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
    with c1:
        c1c = c1.container(border=False)
        with c1c:
            with st.status('validate', expanded=True) as sc1c:
                if st.button("Validation Check"):
                    st.write('validation start')
                    results = test_validate_sqdm(npar)
                    if results:
                        sc1c.update(label='Validation complete...', state='complete')
                    else:
                        sc1c.update(label='Validation failed...', state='error')   

    with c2:
        c2c= c2.container(border=True)
        with c2c:
          with st.status('duplicate', expanded=True) as sc2c:
            st.button("Duplicate Check", use_container_width=True)
            results = test_duplicate_sqdm(npar)
            if results:
                sc2c.update(label='Duplicate complete...', state='complete')
            else:
                sc2c.update(label='Duplicate failed...', state='error')  

    with c3:
        c3c= c3.container(border=True)
        with c3c:
            st.button("Quality Check", use_container_width=True)
            test_quality_sqdm(npar)

    with c4:
        c4c = c4.container(border=True)
        with c4c:
            st.button("Metadata Check", use_container_width=True)
            test_metadata_sqdm(npar)


if __name__=='__main__':
    cn = mp.cpu_count()
    execute(cn // 2)