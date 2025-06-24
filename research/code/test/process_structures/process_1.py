import itertools as it
import pandas as pd

ll = [
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/5442fe69-599e-5cde-a409-bc46145701bc/vcm_s_kf_repr_886x587.jpg",
            "2c8a740d-6e02-4c8f-a87b-e811b040d0f6",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/319dbc8e-2c74-58cd-8f7a-43c1e7447c1d/vcm_s_kf_repr_886x587.jpg",
            "d7901ece-291a-4a6c-b349-fd0a4e9b19f9",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/49ba0a90-d4f1-5420-9bcc-1eaa231656f2/vcm_s_kf_repr_832x624.jpg",
            "80b3fd9f-dfaa-4670-9eac-bec8de1e5ac2",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/916cc21d-6dc2-5369-b994-ae97e411b377/vcm_s_kf_repr_832x624.jpg",
            "287fe7fa-20b7-4a10-9ada-c9b7d3fbc2ae",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/56a82a18-dd40-5ff5-90e8-9122a46a3f94/vcm_s_kf_repr_832x624.jpg",
            "e5eb6cf6-14f1-4400-bc8f-19704767c979",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/07ef46d5-4941-513a-9b4c-7dbc78200519/vcm_s_kf_repr_832x624.jpg",
            "1411b217-baf6-4eec-985e-94c4d13068a6",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/daa1d99c-9ef5-5758-bb69-875e8b530eb7/vcm_s_kf_repr_886x587.jpg",
            "7d1292a6-03ae-49e9-b4cd-ea346df2ebe9",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg",
            "cf627bb3-9d74-45ff-831b-536ade393876",
        ],
    ],
    [
        ["1365323374.0"],
        ["1365222901.0"],
        ["1315180357.0"],
        ["1291580101.0"],
        ["1312911544.0"],
        ["1314075205.0"],
        ["1365223146.0"],
        ["1345132415.0"],
    ],
    [
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        ["(47.610378000000004, -122.200676)", "n/a"],
        [
            "(32.941483, -117.233592)",
            "12327, Caminito Mira del Mar, San Diego, San Diego County, California, 92130, United States",
        ],
        ["(36.887856, -118.555145)", "Fresno County, California, United States"],
        ["(32.941483, -117.233592)", "n/a"],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/5442fe69-599e-5cde-a409-bc46145701bc/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/319dbc8e-2c74-58cd-8f7a-43c1e7447c1d/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/49ba0a90-d4f1-5420-9bcc-1eaa231656f2/vcm_s_kf_repr_832x624.jpg",
            "esha",
            "happy",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/916cc21d-6dc2-5369-b994-ae97e411b377/vcm_s_kf_repr_832x624.jpg",
            "esha, anjali, and a person",
            "angry",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/56a82a18-dd40-5ff5-90e8-9122a46a3f94/vcm_s_kf_repr_832x624.jpg",
            "esha, and a person",
            "sad",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/07ef46d5-4941-513a-9b4c-7dbc78200519/vcm_s_kf_repr_832x624.jpg",
            "esha",
            "happy",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/daa1d99c-9ef5-5758-bb69-875e8b530eb7/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg",
            "",
            "angry",
        ],
    ],
]

#print(ll)

ll = [list(x) for x in zip(*ll)]
lr = [list(it.chain(*item)) for item in ll]
print(lr)
df = pd.DataFrame(lr, columns=['url', 'id', 'ts','latlon', 'location', 'url2','names', 'emotion'])
df =df.drop('url2', axis=1)
print(df.head(5))
d = df.to_dict(orient='records')

print(d)