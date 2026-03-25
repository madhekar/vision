from importlib import metadata

try:
    version = metadata.version("package_name")
    print(f"Package package_name version: {version}")
except metadata.PackageNotFoundError:
    print("Package not found.")