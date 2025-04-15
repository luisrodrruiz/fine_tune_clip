#/FineTuneCLIP



## Prerequisites

What things you need to install the software and how to install them.

tqdm: pip install tqdm

PIL: pip install pillow

pytorch: pip install torch

clip: pip install openai-clip

transformers: pip install transformers

lamini: pip install lamini


## Usage: python fine_tune_clip config.json
## Where config.json stores the configuration for training the model.
# The config.json file must contain a dictionary with two mandatory keys: "training_data" and "dev_data" where each of those keys stores a list of paths to data files.
# Each data file is defined in another .json format. 
{
    "title": "/media/data/pdf_extracts/source/2024-maserati-ghibli-owners-manual
",
    "images_path": "/media/avsr2.0/data/car_manuals/image_captions/2024-maserati
-ghibli-owners-manual/images/",
    "images": [
        {
            "image": "page_0_image_9703.jpg",
            "description": "Location of the vehicle identification number (VIN) on the right-hand front foot platform, visible when the cover is opened."
        },
        {
            "image": "page_12_image_36.png",
            "description": "Location of the vehicle identification number (VIN) on the windshield on the driver's side, visible from the front left center of the dashboard."
        },



```
Examples
```

## Deployment

Add additional notes about how to deploy this on a production system.

## Resources

Add links to external resources for this project, such as CI server, bug tracker, etc.
