    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).major)
    print(torch.cuda.get_device_properties(0).minor)