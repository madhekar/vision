sudo docker build -t zmedia .
sudo docker run -p 8501:8501 zmedia

sudo apt-get clean autoclean
sudo apt-get autoremove --yes
rm -rf /var/lib{apt,dpkg,cache,log}

ERROR: Cannot install -r requirements.txt (line 12), -r requirements.txt (line 15), -r requirements.txt (line 28), -r requirements.txt (line 31), -r requirements.txt (line 6) and Pillow==11.3.0 because these package versions have conflicting dependencies.


ERROR: Cannot install -r requirements.txt (line 10), -r requirements.txt (line 13), -r requirements.txt (line 24), -r requirements.txt (line 27), -r requirements.txt (line 5) and Pillow==11.3.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested Pillow==11.3.0
    deepface 0.0.93 depends on Pillow>=5.2.0
    imagehash 4.3.1 depends on pillow
    matplotlib 3.10.5 depends on pillow>=8
    pyiqa 0.1.13 depends on Pillow
    streamlit 1.39.0 depends on pillow<11 and >=7.1.0

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts

[notice] A new release of pip is available: 23.0.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
The command '/bin/sh -c pip install --no-cache-dir  --default-timeout=900 -r requirements.txt' returned a non-zero code: 1


---

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "spam-eggs"
version = "2020.0.0"
dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'",
]
requires-python = ">=3.8"
authors = [
  {name = "Pradyun Gedam", email = "pradyun@example.com"},
  {name = "Tzu-Ping Chung", email = "tzu-ping@example.com"},
  {name = "Another person"},
  {email = "different.person@example.com"},
]
maintainers = [
  {name = "Brett Cannon", email = "brett@example.com"}
]
description = "Lovely Spam! Wonderful Spam!"
readme = "README.rst"
license = "MIT"
license-files = ["LICEN[CS]E.*"]
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Programming Language :: Python"
]

[project.optional-dependencies]
gui = ["PyQt5"]
cli = [
  "rich",
  "click",
]

[project.urls]
Homepage = "https://example.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/spam.git"
"Bug Tracker" = "https://github.com/me/spam/issues"
Changelog = "https://github.com/me/spam/blob/master/CHANGELOG.md"

[project.scripts]
spam-cli = "spam:main_cli"

[project.gui-scripts]
spam-gui = "spam:main_gui"

[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"