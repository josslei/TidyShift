# This script is provided by Gazebo.
# It is used to download Google Scanned Objects (GSO).
# To download, execute:
# python3 download_GSO.py -o "GoogleResearch" -c "Scanned Objects by Google Research" -d GSO_download
#
# Source: https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
# Ref:
# @online{GazeboFuel-GoogleResearch-Scanned-Objects-by-Google-Research,
#     title={Scanned Objects by Google Research},
#     organization={Open Robotics},
#     date={2022},
#     month={June},
#     day={2},
#     author={GoogleResearch},
#     url={https://fuel.gazebosim.org/1.0/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research},
# }

# Usage
#     python3 download_GSO.py -o <collection_owner> -c <collection_name> -d <output_directory>
#
# Description
#     This script will download all models contained within a collection.
#
import sys, os, json, requests
import getopt

if sys.version_info[0] < 3:
    raise Exception("Python 3 or greater is required. Try running `python3 download_GSO.py`")

collection_name = ""
owner_name = ""
output_dir = ""

# Read options
optlist, args = getopt.getopt(sys.argv[1:], "o:c:d:")

sensor_config_file = ""
private_token = ""
for o, v in optlist:
    if o == "-o":
        owner_name = v.replace(" ", "%20")
    if o == "-c":
        collection_name = v.replace(" ", "%20")
    if o == "-d":
        output_dir = v

if not owner_name:
    print("Error: missing `-o <owner_name>` option")
    quit()

if not collection_name:
    print("Error: missing `-c <collection_name>` option")
    quit()


print(
    "Downloading models from the {}/{} collection.".format(
        owner_name, collection_name.replace("%20", " ")
    )
)

page = 1
count = 0

# The Fuel server URL.
base_url = "https://fuel.gazebosim.org/"

# Fuel server version.
fuel_version = "1.0"

# Path to get the models in the collection
next_url = "/models?page={}&per_page=100&q=collections:{}".format(page, collection_name)

# Path to download a single model in the collection
download_url = base_url + fuel_version + "/{}/models/".format(owner_name)

# Iterate over the pages
while True:
    url = base_url + fuel_version + next_url

    # Get the contents of the current page.
    r = requests.get(url)

    if not r or not r.text:
        break

    # Convert to JSON
    models = json.loads(r.text)

    # Compute the next page's URL
    page = page + 1
    next_url = "/models?page={}&per_page=100&q=collections:{}".format(page, collection_name)

    # Download each model
    for model in models:
        count += 1
        model_name = model["name"]
        print("Downloading (%d) %s" % (count, model_name))
        download = requests.get(download_url + model_name + ".zip", stream=True)
        with open(os.path.join(output_dir, model_name + ".zip"), "wb") as fd:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

print("Done.")
