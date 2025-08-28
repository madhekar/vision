(zm) madhekar@madhekar-UM690:~/work/vision/research/code/test/zm$ uv tree
Resolved 252 packages in 8ms
zm v0.1.0
├── aiofiles v24.1.0
├── aiomultiprocess v0.9.1
├── chardet v4.0.0
├── chromadb v0.6.3
│   ├── bcrypt v4.3.0
│   ├── build v1.3.0
│   │   ├── packaging v24.2
│   │   ├── pyproject-hooks v1.2.0
│   │   └── tomli v2.2.1
│   ├── chroma-hnswlib v0.7.6
│   │   └── numpy v1.26.4
│   ├── fastapi v0.116.1
│   │   ├── pydantic v2.11.7
│   │   │   ├── annotated-types v0.7.0
│   │   │   ├── pydantic-core v2.33.2
│   │   │   │   └── typing-extensions v4.14.1
│   │   │   ├── typing-extensions v4.14.1
│   │   │   └── typing-inspection v0.4.1
│   │   │       └── typing-extensions v4.14.1
│   │   ├── starlette v0.47.2
│   │   │   ├── anyio v4.10.0
│   │   │   │   ├── exceptiongroup v1.3.0
│   │   │   │   │   └── typing-extensions v4.14.1
│   │   │   │   ├── idna v3.10
│   │   │   │   ├── sniffio v1.3.1
│   │   │   │   └── typing-extensions v4.14.1
│   │   │   └── typing-extensions v4.14.1
│   │   └── typing-extensions v4.14.1
│   ├── grpcio v1.74.0
│   ├── httpx v0.28.1
│   │   ├── anyio v4.10.0 (*)
│   │   ├── certifi v2025.8.3
│   │   ├── httpcore v1.0.9
│   │   │   ├── certifi v2025.8.3
│   │   │   └── h11 v0.16.0
│   │   └── idna v3.10
│   ├── importlib-resources v6.5.2
│   ├── kubernetes v33.1.0
│   │   ├── certifi v2025.8.3
│   │   ├── durationpy v0.10
│   │   ├── google-auth v2.40.3
│   │   │   ├── cachetools v5.5.2
│   │   │   ├── pyasn1-modules v0.4.2
│   │   │   │   └── pyasn1 v0.6.1
│   │   │   └── rsa v4.9.1
│   │   │       └── pyasn1 v0.6.1
│   │   ├── oauthlib v3.3.1
│   │   ├── python-dateutil v2.9.0.post0
│   │   │   └── six v1.17.0
│   │   ├── pyyaml v6.0.2
│   │   ├── requests v2.32.5
│   │   │   ├── certifi v2025.8.3
│   │   │   ├── charset-normalizer v3.4.3
│   │   │   ├── idna v3.10
│   │   │   ├── urllib3 v2.5.0
│   │   │   └── pysocks v1.7.1 (extra: socks)
│   │   ├── requests-oauthlib v2.0.0
│   │   │   ├── oauthlib v3.3.1
│   │   │   └── requests v2.32.5 (*)
│   │   ├── six v1.17.0
│   │   ├── urllib3 v2.5.0
│   │   └── websocket-client v1.8.0
│   ├── mmh3 v5.2.0
│   ├── numpy v1.26.4
│   ├── onnxruntime v1.22.1
│   │   ├── coloredlogs v15.0.1
│   │   │   └── humanfriendly v10.0
│   │   ├── flatbuffers v25.2.10
│   │   ├── numpy v1.26.4
│   │   ├── packaging v24.2
│   │   ├── protobuf v4.25.8
│   │   └── sympy v1.14.0
│   │       └── mpmath v1.3.0
│   ├── opentelemetry-api v1.27.0
│   │   ├── deprecated v1.2.18
│   │   │   └── wrapt v1.17.3
│   │   └── importlib-metadata v8.4.0
│   │       └── zipp v3.23.0
│   ├── opentelemetry-exporter-otlp-proto-grpc v1.27.0
│   │   ├── deprecated v1.2.18 (*)
│   │   ├── googleapis-common-protos v1.70.0
│   │   │   └── protobuf v4.25.8
│   │   ├── grpcio v1.74.0
│   │   ├── opentelemetry-api v1.27.0 (*)
│   │   ├── opentelemetry-exporter-otlp-proto-common v1.27.0
│   │   │   └── opentelemetry-proto v1.27.0
│   │   │       └── protobuf v4.25.8
│   │   ├── opentelemetry-proto v1.27.0 (*)
│   │   └── opentelemetry-sdk v1.27.0
│   │       ├── opentelemetry-api v1.27.0 (*)
│   │       ├── opentelemetry-semantic-conventions v0.48b0
│   │       │   ├── deprecated v1.2.18 (*)
│   │       │   └── opentelemetry-api v1.27.0 (*)
│   │       └── typing-extensions v4.14.1
│   ├── opentelemetry-instrumentation-fastapi v0.48b0
│   │   ├── opentelemetry-api v1.27.0 (*)
│   │   ├── opentelemetry-instrumentation v0.48b0
│   │   │   ├── opentelemetry-api v1.27.0 (*)
│   │   │   ├── setuptools v80.9.0
│   │   │   └── wrapt v1.17.3
│   │   ├── opentelemetry-instrumentation-asgi v0.48b0
│   │   │   ├── asgiref v3.9.1
│   │   │   │   └── typing-extensions v4.14.1
│   │   │   ├── opentelemetry-api v1.27.0 (*)
│   │   │   ├── opentelemetry-instrumentation v0.48b0 (*)
│   │   │   ├── opentelemetry-semantic-conventions v0.48b0 (*)
│   │   │   └── opentelemetry-util-http v0.48b0
│   │   ├── opentelemetry-semantic-conventions v0.48b0 (*)
│   │   └── opentelemetry-util-http v0.48b0
│   ├── opentelemetry-sdk v1.27.0 (*)
│   ├── orjson v3.11.2
│   ├── overrides v7.7.0
│   ├── posthog v6.6.0
│   │   ├── backoff v2.2.1
│   │   ├── distro v1.9.0
│   │   ├── python-dateutil v2.9.0.post0 (*)
│   │   ├── requests v2.32.5 (*)
│   │   ├── six v1.17.0
│   │   └── typing-extensions v4.14.1
│   ├── pydantic v2.11.7 (*)
│   ├── pypika v0.48.9
│   ├── pyyaml v6.0.2
│   ├── rich v13.9.4
│   │   ├── markdown-it-py v4.0.0
│   │   │   └── mdurl v0.1.2
│   │   ├── pygments v2.19.2
│   │   └── typing-extensions v4.14.1
│   ├── tenacity v9.1.2
│   ├── tokenizers v0.15.2
│   │   └── huggingface-hub v0.34.4
│   │       ├── filelock v3.19.1
│   │       ├── fsspec v2025.7.0
│   │       ├── hf-xet v1.1.8
│   │       ├── packaging v24.2
│   │       ├── pyyaml v6.0.2
│   │       ├── requests v2.32.5 (*)
│   │       ├── tqdm v4.67.1
│   │       └── typing-extensions v4.14.1
│   ├── tqdm v4.67.1
│   ├── typer v0.16.1
│   │   ├── click v8.2.1
│   │   ├── rich v13.9.4 (*)
│   │   ├── shellingham v1.5.4
│   │   └── typing-extensions v4.14.1
│   ├── typing-extensions v4.14.1
│   └── uvicorn[standard] v0.35.0
│       ├── click v8.2.1
│       ├── h11 v0.16.0
│       ├── typing-extensions v4.14.1
│       ├── httptools v0.6.4 (extra: standard)
│       ├── python-dotenv v1.1.1 (extra: standard)
│       ├── pyyaml v6.0.2 (extra: standard)
│       ├── uvloop v0.21.0 (extra: standard)
│       ├── watchfiles v1.1.0 (extra: standard)
│       │   └── anyio v4.10.0 (*)
│       └── websockets v15.0.1 (extra: standard)
├── country-converter v1.3.1
│   └── pandas v2.3.1
│       ├── numpy v1.26.4
│       ├── python-dateutil v2.9.0.post0 (*)
│       ├── pytz v2025.2
│       └── tzdata v2025.2
├── deepface v0.0.93
│   ├── fire v0.7.1
│   │   └── termcolor v3.1.0
│   ├── flask v3.1.1
│   │   ├── blinker v1.9.0
│   │   ├── click v8.2.1
│   │   ├── itsdangerous v2.2.0
│   │   ├── jinja2 v3.1.6
│   │   │   └── markupsafe v3.0.2
│   │   ├── markupsafe v3.0.2
│   │   └── werkzeug v3.1.3
│   │       └── markupsafe v3.0.2
│   ├── flask-cors v6.0.1
│   │   ├── flask v3.1.1 (*)
│   │   └── werkzeug v3.1.3 (*)
│   ├── gdown v5.2.0
│   │   ├── beautifulsoup4 v4.13.4
│   │   │   ├── soupsieve v2.7
│   │   │   └── typing-extensions v4.14.1
│   │   ├── filelock v3.19.1
│   │   ├── requests[socks] v2.32.5 (*)
│   │   └── tqdm v4.67.1
│   ├── gunicorn v23.0.0
│   │   └── packaging v24.2
│   ├── keras v3.11.2
│   │   ├── absl-py v2.3.1
│   │   ├── h5py v3.14.0
│   │   │   └── numpy v1.26.4
│   │   ├── ml-dtypes v0.3.2
│   │   │   └── numpy v1.26.4
│   │   ├── namex v0.1.0
│   │   ├── numpy v1.26.4
│   │   ├── optree v0.17.0
│   │   │   └── typing-extensions v4.14.1
│   │   ├── packaging v24.2
│   │   └── rich v13.9.4 (*)
│   ├── mtcnn v1.0.0
│   │   ├── joblib v1.4.2
│   │   └── lz4 v4.4.4
│   ├── numpy v1.26.4
│   ├── opencv-python v4.10.0.84
│   │   └── numpy v1.26.4
│   ├── pandas v2.3.1 (*)
│   ├── pillow v10.1.0
│   ├── requests v2.32.5 (*)
│   ├── retina-face v0.0.17
│   │   ├── gdown v5.2.0 (*)
│   │   ├── numpy v1.26.4
│   │   ├── opencv-python v4.10.0.84 (*)
│   │   ├── pillow v10.1.0
│   │   └── tensorflow v2.16.1
│   │       ├── absl-py v2.3.1
│   │       ├── astunparse v1.6.3
│   │       │   ├── six v1.17.0
│   │       │   └── wheel v0.45.1
│   │       ├── flatbuffers v25.2.10
│   │       ├── gast v0.6.0
│   │       ├── google-pasta v0.2.0
│   │       │   └── six v1.17.0
│   │       ├── grpcio v1.74.0
│   │       ├── h5py v3.14.0 (*)
│   │       ├── keras v3.11.2 (*)
│   │       ├── libclang v18.1.1
│   │       ├── ml-dtypes v0.3.2 (*)
│   │       ├── numpy v1.26.4
│   │       ├── opt-einsum v3.4.0
│   │       ├── packaging v24.2
│   │       ├── protobuf v4.25.8
│   │       ├── requests v2.32.5 (*)
│   │       ├── setuptools v80.9.0
│   │       ├── six v1.17.0
│   │       ├── tensorboard v2.16.2
│   │       │   ├── absl-py v2.3.1
│   │       │   ├── grpcio v1.74.0
│   │       │   ├── markdown v3.8.2
│   │       │   ├── numpy v1.26.4
│   │       │   ├── protobuf v4.25.8
│   │       │   ├── setuptools v80.9.0
│   │       │   ├── six v1.17.0
│   │       │   ├── tensorboard-data-server v0.7.2
│   │       │   └── werkzeug v3.1.3 (*)
│   │       ├── tensorflow-io-gcs-filesystem v0.37.1
│   │       ├── termcolor v3.1.0
│   │       ├── typing-extensions v4.14.1
│   │       └── wrapt v1.17.3
│   ├── tensorflow v2.16.1 (*)
│   └── tqdm v4.67.1
├── exifread v3.4.0
├── fastparquet v2024.11.0
│   ├── cramjam v2.11.0
│   ├── fsspec v2025.7.0
│   ├── numpy v1.26.4
│   ├── packaging v24.2
│   └── pandas v2.3.1 (*)
├── folium v0.17.0
│   ├── branca v0.8.1
│   │   └── jinja2 v3.1.6 (*)
│   ├── jinja2 v3.1.6 (*)
│   ├── numpy v1.26.4
│   ├── requests v2.32.5 (*)
│   └── xyzservices v2025.4.0
├── geopy v2.4.1
│   └── geographiclib v2.0
├── gpsphoto v2.2.3
├── imagehash v4.3.1
│   ├── numpy v1.26.4
│   ├── pillow v10.1.0
│   ├── pywavelets v1.8.0
│   │   └── numpy v1.26.4
│   └── scipy v1.15.3
│       └── numpy v1.26.4
├── joblib v1.4.2
├── keras-facenet v0.3.2
│   └── mtcnn v1.0.0 (*)
├── matplotlib v3.10.5
│   ├── contourpy v1.3.2
│   │   └── numpy v1.26.4
│   ├── cycler v0.12.1
│   ├── fonttools v4.59.1
│   ├── kiwisolver v1.4.9
│   ├── numpy v1.26.4
│   ├── packaging v24.2
│   ├── pillow v10.1.0
│   ├── pyparsing v3.2.3
│   └── python-dateutil v2.9.0.post0 (*)
├── mtcnn v1.0.0 (*)
├── numpy v1.26.4
├── open-clip-torch v3.1.0
│   ├── ftfy v6.3.1
│   │   └── wcwidth v0.2.13
│   ├── huggingface-hub v0.34.4 (*)
│   ├── regex v2025.7.34
│   ├── safetensors v0.6.2
│   ├── timm v1.0.19
│   │   ├── huggingface-hub v0.34.4 (*)
│   │   ├── pyyaml v6.0.2
│   │   ├── safetensors v0.6.2
│   │   ├── torch v2.3.1
│   │   │   ├── filelock v3.19.1
│   │   │   ├── fsspec v2025.7.0
│   │   │   ├── jinja2 v3.1.6 (*)
│   │   │   ├── networkx v3.4.2
│   │   │   ├── nvidia-cublas-cu12 v12.1.3.1
│   │   │   ├── nvidia-cuda-cupti-cu12 v12.1.105
│   │   │   ├── nvidia-cuda-nvrtc-cu12 v12.1.105
│   │   │   ├── nvidia-cuda-runtime-cu12 v12.1.105
│   │   │   ├── nvidia-cudnn-cu12 v8.9.2.26
│   │   │   │   └── nvidia-cublas-cu12 v12.1.3.1
│   │   │   ├── nvidia-cufft-cu12 v11.0.2.54
│   │   │   ├── nvidia-curand-cu12 v10.3.2.106
│   │   │   ├── nvidia-cusolver-cu12 v11.4.5.107
│   │   │   │   ├── nvidia-cublas-cu12 v12.1.3.1
│   │   │   │   ├── nvidia-cusparse-cu12 v12.1.0.106
│   │   │   │   │   └── nvidia-nvjitlink-cu12 v12.9.86
│   │   │   │   └── nvidia-nvjitlink-cu12 v12.9.86
│   │   │   ├── nvidia-cusparse-cu12 v12.1.0.106 (*)
│   │   │   ├── nvidia-nccl-cu12 v2.20.5
│   │   │   ├── nvidia-nvtx-cu12 v12.1.105
│   │   │   ├── sympy v1.14.0 (*)
│   │   │   ├── triton v2.3.1
│   │   │   │   └── filelock v3.19.1
│   │   │   └── typing-extensions v4.14.1
│   │   └── torchvision v0.18.1
│   │       ├── numpy v1.26.4
│   │       ├── pillow v10.1.0
│   │       └── torch v2.3.1 (*)
│   ├── torch v2.3.1 (*)
│   ├── torchvision v0.18.1 (*)
│   └── tqdm v4.67.1
├── openai-clip v1.0.1
│   ├── ftfy v6.3.1 (*)
│   ├── regex v2025.7.34
│   └── tqdm v4.67.1
├── opencv-contrib-python v4.11.0.86
│   └── numpy v1.26.4
├── opencv-python v4.10.0.84 (*)
├── opencv-python-headless v4.11.0.86
│   └── numpy v1.26.4
├── pandas v2.3.1 (*)
├── piexif v1.1.3
├── pillow v10.1.0
├── plotly v5.24.1
│   ├── packaging v24.2
│   └── tenacity v9.1.2
├── py3exiv2 v0.12.0
├── pyiqa v0.1.13
│   ├── accelerate v1.10.0
│   │   ├── huggingface-hub v0.34.4 (*)
│   │   ├── numpy v1.26.4
│   │   ├── packaging v24.2
│   │   ├── psutil v7.0.0
│   │   ├── pyyaml v6.0.2
│   │   ├── safetensors v0.6.2
│   │   └── torch v2.3.1 (*)
│   ├── addict v2.4.0
│   ├── bitsandbytes v0.47.0
│   │   ├── numpy v1.26.4
│   │   └── torch v2.3.1 (*)
│   ├── einops v0.8.1
│   ├── facexlib v0.3.0
│   │   ├── filterpy v1.4.5
│   │   │   ├── matplotlib v3.10.5 (*)
│   │   │   ├── numpy v1.26.4
│   │   │   └── scipy v1.15.3 (*)
│   │   ├── numba v0.61.2
│   │   │   ├── llvmlite v0.44.0
│   │   │   └── numpy v1.26.4
│   │   ├── numpy v1.26.4
│   │   ├── opencv-python v4.10.0.84 (*)
│   │   ├── pillow v10.1.0
│   │   ├── scipy v1.15.3 (*)
│   │   ├── torch v2.3.1 (*)
│   │   ├── torchvision v0.18.1 (*)
│   │   └── tqdm v4.67.1
│   ├── future v1.0.0
│   ├── icecream v2.1.7
│   │   ├── asttokens v3.0.0
│   │   ├── colorama v0.4.6
│   │   ├── executing v2.2.0
│   │   └── pygments v2.19.2
│   ├── lmdb v1.7.3
│   ├── numpy v1.26.4
│   ├── openai-clip v1.0.1 (*)
│   ├── opencv-python-headless v4.11.0.86 (*)
│   ├── pandas v2.3.1 (*)
│   ├── pillow v10.1.0
│   ├── pre-commit v4.3.0
│   │   ├── cfgv v3.4.0
│   │   ├── identify v2.6.13
│   │   ├── nodeenv v1.9.1
│   │   ├── pyyaml v6.0.2
│   │   └── virtualenv v20.34.0
│   │       ├── distlib v0.4.0
│   │       ├── filelock v3.19.1
│   │       ├── platformdirs v4.3.8
│   │       └── typing-extensions v4.14.1
│   ├── pytest v8.4.1
│   │   ├── exceptiongroup v1.3.0 (*)
│   │   ├── iniconfig v2.1.0
│   │   ├── packaging v24.2
│   │   ├── pluggy v1.6.0
│   │   ├── pygments v2.19.2
│   │   └── tomli v2.2.1
│   ├── pyyaml v6.0.2
│   ├── requests v2.32.5 (*)
│   ├── scikit-image v0.25.2
│   │   ├── imageio v2.37.0
│   │   │   ├── numpy v1.26.4
│   │   │   └── pillow v10.1.0
│   │   ├── lazy-loader v0.4
│   │   │   └── packaging v24.2
│   │   ├── networkx v3.4.2
│   │   ├── numpy v1.26.4
│   │   ├── packaging v24.2
│   │   ├── pillow v10.1.0
│   │   ├── scipy v1.15.3 (*)
│   │   └── tifffile v2025.5.10
│   │       └── numpy v1.26.4
│   ├── scipy v1.15.3 (*)
│   ├── sentencepiece v0.2.1
│   ├── tensorboard v2.16.2 (*)
│   ├── timm v1.0.19 (*)
│   ├── torch v2.3.1 (*)
│   ├── torchvision v0.18.1 (*)
│   ├── tqdm v4.67.1
│   ├── transformers v4.37.2
│   │   ├── filelock v3.19.1
│   │   ├── huggingface-hub v0.34.4 (*)
│   │   ├── numpy v1.26.4
│   │   ├── packaging v24.2
│   │   ├── pyyaml v6.0.2
│   │   ├── regex v2025.7.34
│   │   ├── requests v2.32.5 (*)
│   │   ├── safetensors v0.6.2
│   │   ├── tokenizers v0.15.2 (*)
│   │   └── tqdm v4.67.1
│   └── yapf v0.43.0
│       ├── platformdirs v4.3.8
│       └── tomli v2.2.1
├── python-exiv2 v0.11.0
│   └── exiv2 v0.17.3
├── pyyaml v6.0.2
├── scikit-learn v1.7.1
│   ├── joblib v1.4.2
│   ├── numpy v1.26.4
│   ├── scipy v1.15.3 (*)
│   └── threadpoolctl v3.6.0
├── streamlit v1.39.0
│   ├── altair v5.5.0
│   │   ├── jinja2 v3.1.6 (*)
│   │   ├── jsonschema v4.25.1
│   │   │   ├── attrs v25.3.0
│   │   │   ├── jsonschema-specifications v2025.4.1
│   │   │   │   └── referencing v0.36.2
│   │   │   │       ├── attrs v25.3.0
│   │   │   │       ├── rpds-py v0.27.0
│   │   │   │       └── typing-extensions v4.14.1
│   │   │   ├── referencing v0.36.2 (*)
│   │   │   └── rpds-py v0.27.0
│   │   ├── narwhals v2.1.2
│   │   ├── packaging v24.2
│   │   └── typing-extensions v4.14.1
│   ├── blinker v1.9.0
│   ├── cachetools v5.5.2
│   ├── click v8.2.1
│   ├── gitpython v3.1.45
│   │   └── gitdb v4.0.12
│   │       └── smmap v5.0.2
│   ├── numpy v1.26.4
│   ├── packaging v24.2
│   ├── pandas v2.3.1 (*)
│   ├── pillow v10.1.0
│   ├── protobuf v4.25.8
│   ├── pyarrow v21.0.0
│   ├── pydeck v0.9.1
│   │   ├── jinja2 v3.1.6 (*)
│   │   └── numpy v1.26.4
│   ├── requests v2.32.5 (*)
│   ├── rich v13.9.4 (*)
│   ├── tenacity v9.1.2
│   ├── toml v0.10.2
│   ├── tornado v6.5.2
│   ├── typing-extensions v4.14.1
│   └── watchdog v5.0.3
├── streamlit-folium v0.22.1
│   ├── branca v0.8.1 (*)
│   ├── folium v0.17.0 (*)
│   ├── jinja2 v3.1.6 (*)
│   └── streamlit v1.39.0 (*)
├── streamlit-image-select v0.6.0
│   └── streamlit v1.39.0 (*)
├── streamlit-tree-select v0.0.5
│   └── streamlit v1.39.0 (*)
├── tensorflow v2.16.1 (*)
├── tensorflow-cpu v2.16.1
│   ├── absl-py v2.3.1
│   ├── astunparse v1.6.3 (*)
│   ├── flatbuffers v25.2.10
│   ├── gast v0.6.0
│   ├── google-pasta v0.2.0 (*)
│   ├── grpcio v1.74.0
│   ├── h5py v3.14.0 (*)
│   ├── keras v3.11.2 (*)
│   ├── libclang v18.1.1
│   ├── ml-dtypes v0.3.2 (*)
│   ├── numpy v1.26.4
│   ├── opt-einsum v3.4.0
│   ├── packaging v24.2
│   ├── protobuf v4.25.8
│   ├── requests v2.32.5 (*)
│   ├── setuptools v80.9.0
│   ├── six v1.17.0
│   ├── tensorboard v2.16.2 (*)
│   ├── tensorflow-io-gcs-filesystem v0.37.1
│   ├── termcolor v3.1.0
│   ├── typing-extensions v4.14.1
│   └── wrapt v1.17.3
├── tf-keras v2.16.0
│   └── tensorflow v2.16.1 (*)
├── torch v2.3.1 (*)
├── torchvision v0.18.1 (*)
└── transformers v4.37.2 (*)
(*) Package tree already displayed
