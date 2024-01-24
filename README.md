# Image inpainting
Trying to perform image inpainting using deep learning methods

### Dataset:
[Places2](https://www.kaggle.com/datasets/nickj26/places2-mit-dataset)
Due to enormous size of the dataset we chose to only operate on subset of it.
Selected categories: campus, lawn, house, hotel-outdoor, park, sky,
residential_neighborhood, highway, driveway

## How to run:
Build docker image:
```
docker build -t inpainting .
```

Run docker container:
```
docker run -p 7860:7860 inpainting
```

Go to [http://localhost:7860](http://localhost:7860)
