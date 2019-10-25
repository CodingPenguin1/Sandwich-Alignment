#!/usr/bin/env python

from google_images_download import google_images_download

imageDownloader = google_images_download.googleimagesdownload()

sandwichQueries = [
    'sandwich', 'sub', 'burrito', 'taco', 'filled doughnut', 'hot dog', 'nontraditional sandwich', 'cookie sandwich',
    'food wrapped in food', 'food inside food', 'ice cream sandwich', 'pop tart', 'filled candy', 'candy bar', 'taco',
    'chicken wrap', 'sandwich wrap', 'non conventional sandwich', 'toaster strudel', 'chicken sandwich', 'blt', 'sandwich with egg',
    'philly cheese steak', 'vegetarian sandwiches', 'oreo', 'klondike bar', 'waffle sandwich', 'breakfast sandwich', 'ice cream taco'
]

for query in sandwichQueries:
    arguments = {'keywords': query,
                    'limit': 100,
                    'format': 'jpg',
                    'thumbnail_only': True,
                    'no_directory': True}
    try:
        imageDownloader.download(arguments)
    except:
        print("Couldn't download images for query {}".format(query))