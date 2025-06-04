import pandas as pd
import itertools as it
ll1 = [
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/5442fe69-599e-5cde-a409-bc46145701bc/vcm_s_kf_repr_886x587.jpg",
            "ba9cfe7b-35ff-421b-8d5f-35f6fa1f46f8",
        ],
        ["1365323374.0"],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/5442fe69-599e-5cde-a409-bc46145701bc/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/319dbc8e-2c74-58cd-8f7a-43c1e7447c1d/vcm_s_kf_repr_886x587.jpg",
            "485eec40-5fd8-41d2-9d08-84dbb1e08695",
        ],
        ["1365222901.0"],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/319dbc8e-2c74-58cd-8f7a-43c1e7447c1d/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/49ba0a90-d4f1-5420-9bcc-1eaa231656f2/vcm_s_kf_repr_832x624.jpg",
            "eee2ecc4-2af0-45e3-81ec-805d64fec752",
        ],
        ["1315180357.0"],
        ["(47.610378000000004, -122.200676)", "n/a"],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/49ba0a90-d4f1-5420-9bcc-1eaa231656f2/vcm_s_kf_repr_832x624.jpg",
            "esha",
            "happy",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/916cc21d-6dc2-5369-b994-ae97e411b377/vcm_s_kf_repr_832x624.jpg",
            "7163a7f9-1877-450d-874a-5bcf479bd222",
        ],
        ["1291580101.0"],
        [
            "(32.941483, -117.233592)",
            "12327, Caminito Mira del Mar, San Diego, San Diego County, California, 92130, United States",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/916cc21d-6dc2-5369-b994-ae97e411b377/vcm_s_kf_repr_832x624.jpg",
            "esha, anjali, and a person",
            "angry",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/56a82a18-dd40-5ff5-90e8-9122a46a3f94/vcm_s_kf_repr_832x624.jpg",
            "7808d342-bd96-49d9-a051-f79a21264356",
        ],
        ["1312911544.0"],
        ["(36.887856, -118.555145)", "Fresno County, California, United States"],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/56a82a18-dd40-5ff5-90e8-9122a46a3f94/vcm_s_kf_repr_832x624.jpg",
            "esha, and a person",
            "sad",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/07ef46d5-4941-513a-9b4c-7dbc78200519/vcm_s_kf_repr_832x624.jpg",
            "a1556ca2-8de8-4c3c-b62e-0a8c3b39e795",
        ],
        ["1314075205.0"],
        [
            "(32.941483, -117.233592)",
            "12327, Caminito Mira del Mar, San Diego, San Diego County, California, 92130, United States",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/07ef46d5-4941-513a-9b4c-7dbc78200519/vcm_s_kf_repr_832x624.jpg",
            "esha",
            "happy",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/daa1d99c-9ef5-5758-bb69-875e8b530eb7/vcm_s_kf_repr_886x587.jpg",
            "d9000878-6cbb-4a55-a675-f9971d2936de",
        ],
        ["1365223146.0"],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/daa1d99c-9ef5-5758-bb69-875e8b530eb7/vcm_s_kf_repr_886x587.jpg",
            "",
            "sad",
        ],
    ],
    [
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg",
            "8ab16832-28fe-4868-ae86-d7d6a50f9b7f",
        ],
        ["1345132415.0"],
        [
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ],
        [
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg",
            "",
            "angry",
        ],
    ],
]

ll = [
    [
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/5442fe69-599e-5cde-a409-bc46145701bc/vcm_s_kf_repr_886x587.jpg",
            "2a470c5a-cbb6-4269-82a4-71f3426e7acb",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/319dbc8e-2c74-58cd-8f7a-43c1e7447c1d/vcm_s_kf_repr_886x587.jpg",
            "a26f6faf-195e-4a09-9d29-b6f4f3b57840",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/49ba0a90-d4f1-5420-9bcc-1eaa231656f2/vcm_s_kf_repr_832x624.jpg",
            "d99054f3-010f-4e95-a788-fb1c44f02c9a",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/916cc21d-6dc2-5369-b994-ae97e411b377/vcm_s_kf_repr_832x624.jpg",
            "def12a35-4f15-45b2-b305-cb1f07c1fc46",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/56a82a18-dd40-5ff5-90e8-9122a46a3f94/vcm_s_kf_repr_832x624.jpg",
            "16a7abbd-04a9-4b91-bb01-8fb48e8d2619",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/07ef46d5-4941-513a-9b4c-7dbc78200519/vcm_s_kf_repr_832x624.jpg",
            "5babfef3-ae5a-4e01-b166-e7fba4c0d735",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/daa1d99c-9ef5-5758-bb69-875e8b530eb7/vcm_s_kf_repr_886x587.jpg",
            "d455f10c-80ff-4342-b6f8-74c322c073c3",
        ),
        (
            "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/19951e8d-2921-566b-b23c-dd08ddfb25de/vcm_s_kf_repr_960x540.jpg",
            "e5b33360-b44a-4815-8f81-17b3f3760d5b",
        ),
    ],
    [
        "1365323374.0",
        "1365222901.0",
        "1315180357.0",
        "1291580101.0",
        "1312911544.0",
        "1314075205.0",
        "1365223146.0",
        "1345132415.0",
    ],
    [
        (
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ),
        (
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ),
        (
            "(47.610378000000004, -122.200676)",
            "FedEx, Main Street, Bellevue, King County, Washington, 98004, United States",
        ),
        (
            "(32.941483, -117.233592)",
            "12327, Caminito Mira del Mar, San Diego, San Diego County, California, 92130, United States",
        ),
        ("(36.887856, -118.555145)", "Fresno County, California, United States"),
        (
            "(32.941483, -117.233592)",
            "12327, Caminito Mira del Mar, San Diego, San Diego County, California, 92130, United States",
        ),
        (
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ),
        (
            "(32.9687, -117.184196)",
            "madhekar residence at carmel vally san diego, california",
        ),
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
def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0] + flatten(data[1:]))
    else:
        return (data,)    
    
def ft(data):
    if isinstance(data, tuple) or isinstance(data, list):
        print('--->', type(data))
        return list(it.chain(*data))
    elif isinstance(data, str):
        print('--->' ,type(data), data)
        return data    
#print(ll)

res = [list(x) for x in zip(*ll1)]

print(res)

# df = pd.DataFrame({'data': res})

# df = df['data'].list.flatten()

#r = [flatten(e) for e in res]

# r = list(it.chain(*res))

# print(r)

# r1 = [it if isinstance(item, tuple) or isinstance(item, list) else item for li in res for item in li for it in item ]

# r2 = [list(it.chain(*li)) for li in res if not isinstance(li, str) ]

# #r2 = {type(item) : item for li in res for item in li}

# r2 = [ft(li) for li in res]

# print(r1)