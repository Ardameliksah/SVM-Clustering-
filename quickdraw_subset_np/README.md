## Data Loading

To read the images and their labels, you can use the `numpy.load` function as shown below:

```python
import numpy as np

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

print(train_images.shape) # (20000, 28, 28)
print(test_images.shape) # (5000, 28, 28)
```

## Labels

The integer labels correspond to the following classes:
- 0: rabbit
- 1: yoga
- 2: hand
- 3: snowman
- 4: motorbike

## Visualize Images

You may visualize the images using the `PIL` library

```python
from PIL import Image

random_image = np.random.randint(0, train_images.shape[0], size=1)[0]
Image.fromarray(train_images[random_image]).show()
```

![Sample Image](sample_image.png)





