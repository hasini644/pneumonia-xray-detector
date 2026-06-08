import urllib.request
import os

samples = {
    "sample_1.jpeg": "https://raw.githubusercontent.com/PerceptiLabs/Chest-X-Rays/master/img/chest_0.jpeg",
    "sample_2.jpeg": "https://raw.githubusercontent.com/PerceptiLabs/Chest-X-Rays/master/img/chest_10.jpeg",
    "sample_3.jpeg": "https://raw.githubusercontent.com/PerceptiLabs/Chest-X-Rays/master/img/chest_20.jpeg"
}

print("Downloading sample X-ray images for testing...")

for name, url in samples.items():
    if not os.path.exists(name):
        try:
            urllib.request.urlretrieve(url, name)
            print(f"Downloaded {name} successfully!")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    else:
        print(f"{name} already exists.")
