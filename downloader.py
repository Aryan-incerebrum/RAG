import os
import requests

"""
From originally reverse engineering the api you can obtain collection downloads.
There isn't a standardized API docs for eLibrarySansad however by tracing it back,
you can obtain it. This had worked roughly on around: 1/8/26
DSpace 7.x, an open-source digital repository platform used globally by libraries, schools,
and organizations to index data.
This proved easy to backtrack as the structure was already known. 
"""

BASE = "https://elibrary.sansad.in/server/api"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

# CHANGE THIS TO WHICHEVER COLLECTION YOU WANT
#these are speeches to the lok sabha
COLLECTION_UUID = "1e2face2-6bd3-4a8b-bec6-355d015759cd"

DOWNLOAD_FOLDER = "eLibrarySansadDownloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

page = 0

while True:

    print(f"\n===== Page {page} =====")

    params = {
        "configuration": "default",
        "scope": COLLECTION_UUID,
        "dsoType": "ITEM",
        "page": page,
        "size": 100
    }

    r = session.get(
        BASE + "/discover/search/objects",
        params=params
    )

    r.raise_for_status()

    data = r.json()

    objects = data["_embedded"]["searchResult"]["_embedded"]["objects"]

    if len(objects) == 0:
        print("Finished.")
        break

    for obj in objects:

        item = obj["_embedded"]["indexableObject"]

        item_uuid = item["uuid"]
        title = item["name"]

        print("\n", title)

        # -------------------------
        # Get bundles
        # -------------------------

        bundles_url = BASE + f"/core/items/{item_uuid}/bundles"

        bundles = session.get(bundles_url).json()["_embedded"]["bundles"]

        original = None

        for bundle in bundles:
            if bundle["name"] == "ORIGINAL":
                original = bundle
                break

        if original is None:
            print("No ORIGINAL bundle.")
            continue

        # -------------------------
        # Get bitstreams
        # -------------------------

        bundle_uuid = original["uuid"]

        bitstreams_url = BASE + f"/core/bundles/{bundle_uuid}/bitstreams"

        bitstreams = session.get(bitstreams_url).json()["_embedded"]["bitstreams"]

        for bitstream in bitstreams:

            filename = bitstream["name"]

            if not filename.lower().endswith(".pdf") or "_hindi" in filename:
                continue

            filepath = os.path.join(DOWNLOAD_FOLDER, filename)

            if os.path.exists(filepath):
                print("Already downloaded:", filename)
                continue

            download_url = bitstream["_links"]["content"]["href"]

            print("Downloading:", filename)

            response = session.get(download_url)

            with open(filepath, "wb") as f:
                f.write(response.content)

    page += 1