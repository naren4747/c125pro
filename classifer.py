import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
from sklearn.metrics import accuracy_score

X=np.load("image.npz")['arr_0']
y=pd.read_csv("labels.csv")["labels"]

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.25)

x_train_scaled=x_train
x_test_scaled=x_test

clasifier=LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train_scaled,y_train)

yprediction=clasifier.predict(x_test_scaled)

accuracy=accuracy_score(y_test,yprediction)
print(accuracy)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clasifier.predict(test_sample)
    return test_pred[0]